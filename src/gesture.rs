use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::types::{FingerState, GestureDetail, GestureKind, GestureMotion, Handedness};

const MIN_CONFIDENCE: f32 = 0.2;
const MOTION_WINDOW: Duration = Duration::from_millis(1_200);

pub struct GestureClassifier {
    motion_tracker: MotionTracker,
}

impl GestureClassifier {
    pub fn new() -> Self {
        Self {
            motion_tracker: MotionTracker::new(),
        }
    }

    pub fn classify(
        &mut self,
        raw_landmarks: &[[f32; 3]],
        projected_landmarks: &[(f32, f32)],
        confidence: f32,
        handedness_score: f32,
        timestamp: Instant,
    ) -> Option<GestureDetail> {
        if confidence < MIN_CONFIDENCE {
            return None;
        }
        if raw_landmarks.len() < 21 || projected_landmarks.len() < 21 {
            return None;
        }

        let (normalized, _hand_span) = normalize_landmarks(raw_landmarks);
        let wrist_px = projected_landmarks.get(0).copied().unwrap_or((0.0, 0.0));
        let span_px = projected_span(projected_landmarks);
        let finger_states = [
            classify_thumb(&normalized),
            classify_finger(&normalized, [5, 6, 7, 8]),
            classify_finger(&normalized, [9, 10, 11, 12]),
            classify_finger(&normalized, [13, 14, 15, 16]),
            classify_finger(&normalized, [17, 18, 19, 20]),
        ];

        let handedness = handedness_from_score(handedness_score);
        let primary = detect_primary_gesture(&normalized, &finger_states);
        let secondary = detect_secondary(&finger_states, &normalized, primary);
        let motion = self
            .motion_tracker
            .update(wrist_px, span_px, timestamp, primary);

        Some(GestureDetail {
            primary,
            secondary,
            handedness,
            finger_states,
            motion,
        })
    }
}

fn handedness_from_score(score: f32) -> Handedness {
    if score >= 0.5 {
        Handedness::Right
    } else if score > 0.0 {
        Handedness::Left
    } else {
        Handedness::Unknown
    }
}

fn normalize_landmarks(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for [x, y, _z] in points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    let span = (max_x - min_x).max(max_y - min_y).max(1e-3);
    let normalized = points
        .iter()
        .map(|[x, y, z]| [(*x - min_x) / span, (*y - min_y) / span, *z / span])
        .collect();

    (normalized, span)
}

fn projected_span(points: &[(f32, f32)]) -> f32 {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (max_x - min_x).max(max_y - min_y).max(1.0)
}

fn classify_finger(points: &[[f32; 3]], idx: [usize; 4]) -> FingerState {
    let wrist = points[0];
    let mcp = points[idx[0]];
    let pip = points[idx[1]];
    let dip = points[idx[2]];
    let tip = points[idx[3]];

    let dist_tip = distance3(tip, wrist);
    let dist_pip = distance3(pip, wrist);
    let dist_mcp = distance3(mcp, wrist);

    let straightness = average_straightness(sub(pip, mcp), sub(dip, pip), sub(tip, dip));

    let extension = dist_tip - dist_pip;
    let reach = dist_tip - dist_mcp;

    if extension > 0.18 && straightness > 0.45 && reach > 0.08 {
        FingerState::Extended
    } else if extension < 0.08 || straightness < 0.18 || reach < 0.05 {
        FingerState::Folded
    } else {
        FingerState::HalfBent
    }
}

fn classify_thumb(points: &[[f32; 3]]) -> FingerState {
    let wrist = points[0];
    let mcp = points[1];
    let ip = points[2];
    let tip = points[4];
    let index_mcp = points[5];
    let pinky_mcp = points[17];

    let dist_tip_wrist = distance3(tip, wrist);
    let dist_tip_index = distance3(tip, index_mcp);
    let dist_tip_pinky = distance3(tip, pinky_mcp);
    let straightness = average_straightness(sub(ip, mcp), sub(tip, ip), sub(tip, ip));

    let spread = dist_tip_index.min(dist_tip_pinky);

    if spread < 0.16 && straightness < 0.25 {
        FingerState::Folded
    } else if dist_tip_wrist > 0.35 && straightness > 0.35 {
        FingerState::Extended
    } else {
        FingerState::HalfBent
    }
}

fn detect_primary_gesture(points: &[[f32; 3]], finger_states: &[FingerState; 5]) -> GestureKind {
    let extended_count = finger_states
        .iter()
        .filter(|s| matches!(s, FingerState::Extended))
        .count();
    let folded_count = finger_states
        .iter()
        .filter(|s| matches!(s, FingerState::Folded))
        .count();

    let thumb = finger_states[0];
    let index = finger_states[1];
    let middle = finger_states[2];
    let ring = finger_states[3];
    let pinky = finger_states[4];

    let thumb_index_gap = distance3(points[4], points[8]);
    let thumb_middle_gap = distance3(points[4], points[12]);
    let wrist_y = points[0][1];
    let thumb_tip_y = points[4][1];

    // Finger heart: thumb + index very close, both half-bent, other fingers mostly folded, tips aligned.
    let finger_heart = thumb_index_gap < 0.08
        && folded_count >= 3
        && matches!(index, FingerState::HalfBent | FingerState::Folded)
        && matches!(thumb, FingerState::HalfBent | FingerState::Folded)
        && (points[4][1] - points[8][1]).abs() < 0.08;

    // Kneading/pinch: allow thumb-index or thumb-middle pairing; non-participating fingers not extended.
    let pinch_with_index = thumb_index_gap < 0.12
        && matches!(middle, FingerState::Folded | FingerState::HalfBent)
        && matches!(ring, FingerState::Folded | FingerState::HalfBent)
        && matches!(pinky, FingerState::Folded | FingerState::HalfBent);
    let pinch_with_middle = thumb_middle_gap < 0.12
        && matches!(index, FingerState::Folded | FingerState::HalfBent)
        && matches!(ring, FingerState::Folded | FingerState::HalfBent)
        && matches!(pinky, FingerState::Folded | FingerState::HalfBent);
    let pinch_like = pinch_with_index || pinch_with_middle;
    let ok_like = thumb_index_gap < 0.18
        && (middle == FingerState::Extended || ring == FingerState::Extended);
    let ilove = matches!(thumb, FingerState::Extended | FingerState::HalfBent)
        && index == FingerState::Extended
        && middle != FingerState::Extended
        && ring != FingerState::Extended
        && pinky == FingerState::Extended;
    let rock = index == FingerState::Extended
        && pinky == FingerState::Extended
        && middle != FingerState::Extended
        && ring != FingerState::Extended
        && thumb == FingerState::Folded;
    let victory = index == FingerState::Extended
        && middle == FingerState::Extended
        && ring != FingerState::Extended
        && pinky != FingerState::Extended;
    let point = index == FingerState::Extended
        && middle != FingerState::Extended
        && ring != FingerState::Extended
        && pinky != FingerState::Extended;
    let three = index == FingerState::Extended
        && middle == FingerState::Extended
        && ring == FingerState::Extended
        && pinky != FingerState::Extended;
    let four = extended_count >= 4 && thumb != FingerState::Extended;
    let fist = folded_count >= 4;
    let open_palm = extended_count >= 4;

    let thumb_up =
        thumb == FingerState::Extended && folded_count >= 3 && thumb_tip_y + 0.08 < wrist_y;
    let thumb_down =
        thumb == FingerState::Extended && folded_count >= 3 && thumb_tip_y > wrist_y + 0.08;

    if finger_heart {
        GestureKind::FingerHeart
    } else if pinch_like {
        GestureKind::Pinch
    } else if ok_like {
        GestureKind::Ok
    } else if ilove {
        GestureKind::ILoveYou
    } else if rock {
        GestureKind::Rock
    } else if victory {
        GestureKind::Victory
    } else if point {
        GestureKind::Point
    } else if thumb_up {
        GestureKind::ThumbUp
    } else if thumb_down {
        GestureKind::ThumbDown
    } else if fist {
        GestureKind::Fist
    } else if four {
        GestureKind::Four
    } else if open_palm {
        GestureKind::OpenPalm
    } else if three {
        GestureKind::Three
    } else {
        GestureKind::Unknown
    }
}

fn detect_secondary(
    finger_states: &[FingerState; 5],
    points: &[[f32; 3]],
    primary: GestureKind,
) -> Option<GestureKind> {
    if primary != GestureKind::Unknown {
        return None;
    }

    let extended_count = finger_states
        .iter()
        .filter(|s| matches!(s, FingerState::Extended))
        .count();
    let folded_count = finger_states
        .iter()
        .filter(|s| matches!(s, FingerState::Folded))
        .count();

    if extended_count >= 4 {
        Some(GestureKind::OpenPalm)
    } else if folded_count >= 4 {
        Some(GestureKind::Fist)
    } else if distance3(points[4], points[8]).min(distance3(points[4], points[12])) < 0.14 {
        Some(GestureKind::Pinch)
    } else {
        None
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn average_straightness(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = dot(normalize(a), normalize(b));
    let bc = dot(normalize(b), normalize(c));
    ((ab + bc) / 2.0).clamp(-1.0, 1.0)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-5 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[derive(Clone)]
struct MotionSample {
    time: Instant,
    x: f32,
    y: f32,
    span: f32,
}

struct MotionTracker {
    history: VecDeque<MotionSample>,
}

impl MotionTracker {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
        }
    }

    fn update(
        &mut self,
        point: (f32, f32),
        span: f32,
        now: Instant,
        primary: GestureKind,
    ) -> GestureMotion {
        self.history.push_back(MotionSample {
            time: now,
            x: point.0,
            y: point.1,
            span: span.max(1.0),
        });

        while let Some(front) = self.history.front() {
            if now.duration_since(front.time) > MOTION_WINDOW {
                self.history.pop_front();
            } else {
                break;
            }
        }

        if self.history.len() < 3 {
            return GestureMotion::Steady;
        }

        let avg_span =
            self.history.iter().map(|s| s.span).sum::<f32>() / (self.history.len() as f32);
        let norm = avg_span.max(1.0);

        let (min_x, max_x, min_y, max_y) =
            self.history
                .iter()
                .fold((f32::MAX, f32::MIN, f32::MAX, f32::MIN), |acc, s| {
                    (
                        acc.0.min(s.x),
                        acc.1.max(s.x),
                        acc.2.min(s.y),
                        acc.3.max(s.y),
                    )
                });

        let span_x = (max_x - min_x) / norm;
        let span_y = (max_y - min_y) / norm;

        let samples: Vec<MotionSample> = self.history.iter().cloned().collect();

        let direction_changes_x = direction_changes(&samples, |s| s.x, norm * 0.08);
        let direction_changes_y = direction_changes(&samples, |s| s.y, norm * 0.08);

        let is_open_palm = matches!(
            primary,
            GestureKind::OpenPalm | GestureKind::Four | GestureKind::Unknown
        );

        if span_x > 0.55 && direction_changes_x >= 2 && is_open_palm {
            GestureMotion::Fanning
        } else if span_y > 0.55 && direction_changes_y >= 2 {
            GestureMotion::VerticalWave
        } else if span_x > 0.25 || span_y > 0.25 {
            GestureMotion::Moving
        } else {
            GestureMotion::Steady
        }
    }
}

fn direction_changes<F>(samples: &[MotionSample], select: F, min_step: f32) -> usize
where
    F: Fn(&MotionSample) -> f32,
{
    let mut changes = 0;
    let mut last_sign = 0i8;

    for pair in samples.windows(2) {
        let delta = select(&pair[1]) - select(&pair[0]);
        if delta.abs() < min_step {
            continue;
        }
        let sign = if delta > 0.0 { 1 } else { -1 };
        if last_sign != 0 && sign != last_sign {
            changes += 1;
        }
        last_sign = sign;
    }

    changes
}
