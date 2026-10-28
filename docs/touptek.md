# ToupTek E3ISPM20000KPA (0547:1122, USB2 mode 0547:1123) — protocol notes

Worked out by static analysis of the vendor SDK library (libmiicam.so from
miicamsdk 20250709, linux/x64; the library was never run), then checked on
the hardware. Addresses below are file virtual addresses in that library.
Implemented in `src/touptek.rs`.

Sensor: Sony IMX183 (chip id reg 0xFFFF reads 0x183A). Model flags 0x81442041
(CMOS, USB30, ISP, BUFFER, BLACKLEVEL, RAW8, RAW12).
Resolutions: 5440x3648@15, 2736x1824@50 (bin2), 1824x1216@60 (bin3).

## Control transfers (EP0, vendor, device)
setup struct in lib: [dir(0=OUT,1=IN)][bRequest][wValue][wIndex]; timeout 2000 ms.
Key: k = session key (u16); kr = ror16(k, 4). Encrypted requests XOR wValue and wIndex with kr.
With k = 0 everything is plain.

- 0x16 IN, wValue = k (plain), 2 bytes: set key; reply[0] == 0x08 means OK (retry 200 ms, 2 s).  "bCheckSelf"
- challenge: OUT req in 0x41..0x60 (0x41 default) 16 bytes; IN req in 0x61..0x80 except 0x77, 16 bytes.
  Host-side anti-clone check; we can ignore the response.  "bCheckCRC"
- 0x01 OUT wValue=v wIndex=0x0F: legacy init. Open: 1,0,1. Start stream: 3. "bLegacyInit"
- 0x0B reg write: wValue = val ^ kr, wIndex = reg ^ kr. On this device (bcdDevice 0x0002) IN with 1 byte reply (0x08).
- 0x0A reg read: IN, wValue = kr, wIndex = reg ^ kr, 2 bytes big-endian. Issued twice, second used.
- 0x20 IN EEPROM read: wValue = offset (8 KiB, wraps), up to 4096/req.
- 0x10 IN 1 byte (read after chip id in OpenUSB) -> 0x33 on this unit.
- AVOID: 0x21 (EEPROM write), 0x31/0xA0/0xD9 (flash), unknown OUT requests.

## FPGA registers -> Sony sensor
reg 0x1000+p, value (a<<8)|d  => sensor register (0x3000 + p*0x100 + a) = d.
Register tables: u16 pairs (reg, val); reg 0xFFFF => sleep val ms (0 => yield).

## Open (OpenUSB, model specific @0x1974490)
bOpenUSB: alloc bulk buffers; key; checkSelf; checkCRC; legacy init 1,0,1;
then poll reg 0xFFFF until 0x183A (100 ms, 2 s); read req 0x10; read reg 0xFEFF.

## Start (startStep @0x1798290 -> startcamera @0x1974700)
1. table A: sleep30, S3058=37 S305a=2b S3312=11 S3355=00 S3381=00 S352e=02 S3530..3535=0b S353f=1d S3541=1d
   S3545=00 S3549=02 S354b=00 S3555=02 S3563=05 S35a4=00 S35a5=07 S35aa=00 S35d1=16 S35d2=15 S35d3=14 S365c=01 S365e=01
2. reg 0x0200 = bitdepth (0 = 8-bit, 1 = 12-bit)   3. reg 0x1000 = 0x0200 (S3002=00)
4. mode (res index): 0 full 5440x3648: S3003=00 S3004=00 S3005=03 S3006=10 S3007=00; fw>=0x204: a200=0x550 a600=0x25 (old: 0x220,0x0f); a400=0xe40 a800=0x29 2000=0
   1 bin2 2736x1824: S3003=22 S3004=19 S3005=11 S3006=50 S3007=00; fw>=0x204: a200=0x2ac a600=0x10 a800=0x0f (old 0x1c8,0x0b,0x10); a400=0x720 2000=1
   2 bin3 1824x1216: S3003=33 S3004=2e S3005=18 S3006=10 S3007=00 a200=0x1c8 a400=0x4c0 a600=0x0b a800=0x0f 2000=2
   then speed: reg 0x8000 = HMAX; then exposure (below)
5. table B: S3008 S300d S300e S300f S3010 S3011 S301a S306f S3070 S3071 S3072 = 0, S3000=02, sleep10, S3000=00, sleep30, S3001=11
6. reg 0x0a00 = 0, sleep 10, reg 0x0a00 = 0xffff
7. ISP: f200 = IspEn, fa00 = 1, f400 = 8 (bayer), f600 = 2 (debayer). IspEn=1 gives zero/variable frames (needs WB setup); use 0 for raw.
8. fw in [0x201,0x1000): ee00 = 1; fw in [0x600,0x1000): be00 = 0x0e.   (this unit fw = reg 0xFEFF = 0x0502)
9. OUT 0x01 wValue=3 wIndex=0x0F: stream on.  (wValue 0: off)

HMAX (reg 0x8000) [mode][speed], 8-bit USB3 (12-bit: second value):
 mode0: 1480/2960, 1000/2000, 850/1700;  mode1: 520/860, 340/580, 254/500;  mode2: 400, 300, 210/260
 USB2 (0x1123): mode0 0x397c/0x26ce/0x1cbe, mode1 0xe70/0x9be/0x738, mode2 0x660/0x44c/0x330 (×2 for 12-bit)
Exposure (us): L = max(minL, (us*48 - off)/HMAX); off/maxL/VMAX/minL: m0 0xd1/0xe81/0xe90/4, m1 0x9d/0xe72/0xe81/4, m2 0x87/0xe91/0xea0/8
 L < maxL: SHS = VMAX - L, else SHS = 0x0f, VMAX = L + 0x0f.  Writes: S300b=SHS lo, S300c=SHS hi, reg4000=VMAX>>16, reg5000=VMAX&0xffff
Gain (percent g, 100 = 1x): v = -204800/g; S3009 = v & 0xff, S300a = (v>>8) & 7
Black level: bc00 = v/3, bd00 = v/3

### Frames
Bulk EP 0x81. One frame = w*h bytes (8-bit RGGB) + 60-byte trailer; trailer last 12 bytes = u64 timestamp, u32 frame no.
Frame ends with a short packet. Must keep several transfers queued, otherwise the FPGA drops data mid-frame.
Measured: 15.1 / 50.9 / 61.1 fps (modes 0/1/2, speed 2) on USB3, no losses with 12 × 1 MiB transfers queued.
Black level in 8-bit frames: 12.

### Stop
IN 0x17, 2 bytes (firmware with bcdDevice < 5), then cancel the transfers.

### EEPROM (req 0x20)
u32 n, u32 ?, then a bzip2 stream up to n: 333 records (u16 id, u16 count, count × u16) — per-unit
calibration, not needed for streaming. At n: NUL-terminated serial number ("TP" + date + hash).

### Not done yet
12-bit output (reg 0x0200 = 1, HMAX doubled; frame and pixel packing unverified), onboard ISP
(white balance etc.), black level registers, ROI, trigger.
