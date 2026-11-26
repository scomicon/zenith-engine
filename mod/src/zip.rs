use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_INFLATED_BYTES: usize = 8 << 20;
const MAX_CENTRAL_DIR_BYTES: usize = 16 << 20;

/// Inflates a raw DEFLATE stream (RFC 1951), as `.hshader` and `.zip` packs store their files;
/// `None` for a broken stream or one past 8 MiB.
pub fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    struct Bits<'a> {
        data: &'a [u8],
        pos: usize,
        buf: u32,
        count: u32,
    }
    impl Bits<'_> {
        fn need(&mut self, n: u32) -> Option<u32> {
            while self.count < n {
                let b = *self.data.get(self.pos)?;
                self.pos += 1;
                self.buf |= u32::from(b) << self.count;
                self.count += 8;
            }
            let v = self.buf & ((1u32 << n) - 1);
            self.buf >>= n;
            self.count -= n;
            Some(v)
        }
    }

    struct Huffman {
        counts: [u16; 16],
        symbols: Vec<u16>,
    }

    fn huffman(lengths: &[u8]) -> Option<Huffman> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            *counts.get_mut(usize::from(l))? += 1;
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len].checked_add(counts[len])?;
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &l) in lengths.iter().enumerate() {
            if l != 0 {
                let slot = offsets.get_mut(usize::from(l))?;
                let target = symbols.get_mut(usize::from(*slot))?;
                *target = symbol as u16;
                *slot = slot.checked_add(1)?;
            }
        }
        Some(Huffman { counts, symbols })
    }

    fn decode(bits: &mut Bits<'_>, h: &Huffman) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= bits.need(1)? as i32;
            let count = i32::from(h.counts[len]);
            if code - count < first {
                return h.symbols.get((index + code - first) as usize).copied();
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        None
    }

    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227,
        258,
    ];
    const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
        6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
    ];
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

    let mut bits = Bits {
        data,
        pos: 0,
        buf: 0,
        count: 0,
    };
    let mut out = Vec::new();
    loop {
        let last = bits.need(1)?;
        let kind = bits.need(2)?;
        let (lengths, distances) = match kind {
            0 => {
                bits.buf = 0;
                bits.count = 0;
                let at = bits.pos;
                let head = data.get(at..at.checked_add(4)?)?;
                let len_u16 = u16::from_le_bytes([head[0], head[1]]);
                let nlen_u16 = u16::from_le_bytes([head[2], head[3]]);
                if len_u16 != !nlen_u16 {
                    return None;
                }
                let len = usize::from(len_u16);
                if out.len() + len > MAX_INFLATED_BYTES {
                    return None;
                }
                let end = at + 4 + len;
                out.extend_from_slice(data.get(at + 4..end)?);
                bits.pos = end;
                if last == 1 {
                    break;
                }
                continue;
            }
            1 => {
                let mut l = [8u8; 288];
                l[144..256].fill(9);
                l[256..280].fill(7);
                (huffman(&l)?, huffman(&[5u8; 30])?)
            }
            2 => {
                let nlen = bits.need(5)? as usize + 257;
                let ndist = bits.need(5)? as usize + 1;
                let ncode = bits.need(4)? as usize + 4;
                let mut code_lengths = [0u8; 19];
                for &i in &ORDER[..ncode] {
                    code_lengths[i] = bits.need(3)? as u8;
                }
                let code = huffman(&code_lengths)?;
                let mut l = Vec::with_capacity(nlen + ndist);
                while l.len() < nlen + ndist {
                    let sym = decode(&mut bits, &code)?;
                    match sym {
                        0..=15 => l.push(sym as u8),
                        16 => {
                            let previous = *l.last()?;
                            let n = 3 + bits.need(2)?;
                            l.extend(std::iter::repeat_n(previous, n as usize));
                        }
                        17 => {
                            let n = 3 + bits.need(3)?;
                            l.extend(std::iter::repeat_n(0, n as usize));
                        }
                        _ => {
                            let n = 11 + bits.need(7)?;
                            l.extend(std::iter::repeat_n(0, n as usize));
                        }
                    }
                }
                if l.len() != nlen + ndist {
                    return None;
                }
                (huffman(&l[..nlen])?, huffman(&l[nlen..])?)
            }
            _ => return None,
        };
        loop {
            let sym = usize::from(decode(&mut bits, &lengths)?);
            if sym < 256 {
                out.push(sym as u8);
            } else if sym == 256 {
                break;
            } else {
                let i = sym - 257;
                let len = usize::from(*LEN_BASE.get(i)?) + bits.need(u32::from(LEN_EXTRA[i]))? as usize;
                let d = usize::from(decode(&mut bits, &distances)?);
                let dist = usize::from(*DIST_BASE.get(d)?) + bits.need(u32::from(DIST_EXTRA[d]))? as usize;
                if dist > out.len() || out.len() + len > MAX_INFLATED_BYTES {
                    return None;
                }
                for _ in 0..len {
                    out.push(out[out.len() - dist]);
                }
            }
            if out.len() > MAX_INFLATED_BYTES {
                return None;
            }
        }
        if last == 1 {
            break;
        }
    }
    Some(out)
}

fn u16_at(bytes: &[u8], i: usize) -> Option<usize> {
    Some(usize::from(u16::from_le_bytes(bytes.get(i..i + 2)?.try_into().ok()?)))
}

fn u32_at(bytes: &[u8], i: usize) -> Option<usize> {
    Some(u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?) as usize)
}

/// Reads the entry matching `wanted` from a ZIP archive without reading the whole file into memory.
pub fn zip_file_entry(path: &Path, wanted: impl Fn(&str) -> bool) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    let file_len = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if file_len < 22 {
        return None;
    }

    let tail_len = file_len.min(65_557);
    let tail_start = file_len - tail_len;
    file.seek(SeekFrom::Start(tail_start as u64)).ok()?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail).ok()?;

    let end = (0..=tail_len - 22).rev().find(|&i| {
        tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]
            && u16_at(&tail, i + 20).is_some_and(|comment_len| tail_start + i + 22 + comment_len == file_len)
    })?;
    let entries = u16_at(&tail, end + 10)?;
    let cd_size = u32_at(&tail, end + 12)?;
    let cd_offset = u32_at(&tail, end + 16)?;
    if cd_size > MAX_CENTRAL_DIR_BYTES || cd_offset.checked_add(cd_size)? > file_len {
        return None;
    }

    file.seek(SeekFrom::Start(cd_offset as u64)).ok()?;
    let mut cd = vec![0u8; cd_size];
    file.read_exact(&mut cd).ok()?;

    let mut at = 0usize;
    for _ in 0..entries {
        if cd.get(at..at + 4)? != [0x50, 0x4b, 0x01, 0x02] {
            return None;
        }
        let method = u16_at(&cd, at + 10)?;
        let size = u32_at(&cd, at + 20)?;
        let name_len = u16_at(&cd, at + 28)?;
        let extra_len = u16_at(&cd, at + 30)?;
        let comment_len = u16_at(&cd, at + 32)?;
        let local = u32_at(&cd, at + 42)?;
        let name = std::str::from_utf8(cd.get(at + 46..at + 46 + name_len)?).ok()?;
        if wanted(name) {
            if size > MAX_INFLATED_BYTES || local.checked_add(30)? > file_len {
                return None;
            }
            file.seek(SeekFrom::Start(local as u64)).ok()?;
            let mut lfh = [0u8; 30];
            file.read_exact(&mut lfh).ok()?;
            if lfh[..4] != [0x50, 0x4b, 0x03, 0x04] {
                return None;
            }
            let data_start = local
                .checked_add(30)?
                .checked_add(u16_at(&lfh, 26)?)?
                .checked_add(u16_at(&lfh, 28)?)?;
            if data_start.checked_add(size)? > file_len {
                return None;
            }
            file.seek(SeekFrom::Start(data_start as u64)).ok()?;
            let mut data = vec![0u8; size];
            file.read_exact(&mut data).ok()?;
            return match method {
                0 => Some(data),
                8 => inflate(&data),
                _ => None,
            };
        }
        at = at.checked_add(46 + name_len + extra_len + comment_len)?;
    }
    None
}
