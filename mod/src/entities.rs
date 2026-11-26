use std::collections::HashMap;

pub const MAX_ENTITIES: usize = 64;
pub const MAX_ENTITY_PARTS: usize = 448;
pub type EntityCandidate = (f32, [f32; 3], [f32; 3], usize, usize);

/// Reusable scratch buffers for extracting entity/mob bounding boxes each frame with zero heap
/// churn in steady state.
#[derive(Default)]
pub struct EntityScratch {
    candidates: Vec<EntityCandidate>,
    pub pos: Vec<[f32; 3]>,
    parent: Vec<usize>,
    seen: HashMap<[i32; 3], usize>,
    boxes: HashMap<usize, ([f32; 3], [f32; 3])>,
    parts: Vec<([f32; 3], [f32; 3])>,
    pub entity_bytes: Vec<u8>,
    pub part_bytes: Vec<u8>,
}

impl EntityScratch {
    pub fn begin(&mut self) {
        self.candidates.clear();
        self.entity_bytes.clear();
        self.part_bytes.clear();
    }

    pub fn push_candidate(&mut self, dist2: f32, lo: [f32; 3], hi: [f32; 3], start: usize, end: usize) {
        self.candidates.push((dist2, lo, hi, start, end));
    }

    pub fn sort_and_truncate(&mut self) -> &[EntityCandidate] {
        if self.candidates.len() > MAX_ENTITIES {
            self.candidates.select_nth_unstable_by(MAX_ENTITIES, |a, b| a.0.total_cmp(&b.0));
            self.candidates.truncate(MAX_ENTITIES);
        }
        &self.candidates
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn candidate(&self, idx: usize) -> EntityCandidate {
        self.candidates[idx]
    }

    /// Groups the quads in `self.pos` into connected cuboid parts (legs, body, head...).
    pub fn compute_parts(&mut self) {
        self.parts.clear();
        let len = self.pos.len();
        if len == 0 || !len.is_multiple_of(4) || len > 8192 {
            return;
        }
        let quads = len / 4;
        self.parent.clear();
        self.parent.extend(0..quads);
        self.seen.clear();

        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }

        for (k, v) in self.pos.iter().enumerate() {
            let key = v.map(|c| (c * 256.0).round() as i32);
            let q = k / 4;
            match self.seen.get(&key) {
                Some(&other) => {
                    let a = root(&mut self.parent, q);
                    let b = root(&mut self.parent, other);
                    if a != b {
                        self.parent[a] = b;
                    }
                }
                None => {
                    self.seen.insert(key, q);
                }
            }
        }

        self.boxes.clear();
        for q in 0..quads {
            let r = root(&mut self.parent, q);
            let e = self.boxes.entry(r).or_insert(([f32::MAX; 3], [f32::MIN; 3]));
            for v in &self.pos[q * 4..q * 4 + 4] {
                for c in 0..3 {
                    e.0[c] = e.0[c].min(v[c]);
                    e.1[c] = e.1[c].max(v[c]);
                }
            }
        }

        for &(mut lo, mut hi) in self.boxes.values() {
            for c in 0..3 {
                if hi[c] - lo[c] < 0.02 {
                    let m = (hi[c] + lo[c]) * 0.5;
                    lo[c] = m - 0.01;
                    hi[c] = m + 0.01;
                }
            }
            self.parts.push((lo, hi));
        }
    }

    pub fn append_entity(&mut self, lo: [f32; 3], hi: [f32; 3], remaining_after: usize, total_parts: &mut usize) {
        let count = if self.parts.is_empty()
            || self.parts.len() > 32
            || *total_parts + self.parts.len() + remaining_after > MAX_ENTITY_PARTS
        {
            self.parts.clear();
            self.parts.push((lo, hi));
            1
        } else {
            self.parts.len()
        };
        for f in [
            lo[0],
            lo[1],
            lo[2],
            *total_parts as f32,
            hi[0],
            hi[1],
            hi[2],
            count as f32,
        ] {
            self.entity_bytes.extend_from_slice(&f32::to_ne_bytes(f));
        }
        for &(plo, phi) in &self.parts {
            for f in [plo[0], plo[1], plo[2], 0.0, phi[0], phi[1], phi[2], 0.0] {
                self.part_bytes.extend_from_slice(&f32::to_ne_bytes(f));
            }
        }
        *total_parts += count;
    }
}
