use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Frame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub timestamp: Instant,
}

#[derive(Clone, Debug)]
pub struct GestureResult {
    pub label: String,
    pub confidence: f32,
    #[allow(dead_code)]
    pub timestamp: Instant,
    pub landmarks: Option<Vec<(f32, f32)>>,
    pub detail: Option<GestureDetail>,
    pub palm_regions: Vec<PalmRegion>,
}

#[derive(Clone, Debug)]
pub struct PalmRegion {
    pub bbox: [f32; 4],
    pub landmarks: Vec<(f32, f32)>,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct RecognizedFrame {
    pub frame: Frame,
    pub result: GestureResult,
}

impl GestureResult {
    #[allow(dead_code)]
    pub fn display_text(&self) -> String {
        if let Some(detail) = &self.detail {
            format!(
                "{}{} ({:.0}%)",
                detail.primary.emoji(),
                detail.primary.display_name(),
                self.confidence * 100.0
            )
        } else {
            format!("{} ({:.0}%)", self.label, self.confidence * 100.0)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
    Unknown,
}

impl Handedness {
    pub fn label(&self) -> &'static str {
        match self {
            Handedness::Left => "左手",
            Handedness::Right => "右手",
            Handedness::Unknown => "未知",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FingerState {
    Extended,
    HalfBent,
    Folded,
}

impl FingerState {
    pub fn label(&self) -> &'static str {
        match self {
            FingerState::Extended => "伸直",
            FingerState::HalfBent => "半弯",
            FingerState::Folded => "弯曲",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureKind {
    OpenPalm,
    Fist,
    Point,
    Victory,
    Three,
    Four,
    ThumbUp,
    ThumbDown,
    Ok,
    Pinch,
    FingerHeart,
    ILoveYou,
    Rock,
    Unknown,
}

impl GestureKind {
    pub fn display_name(&self) -> &'static str {
        match self {
            GestureKind::OpenPalm => "张开手掌",
            GestureKind::Fist => "握拳",
            GestureKind::Point => "指向",
            GestureKind::Victory => "剪刀手",
            GestureKind::Three => "三指",
            GestureKind::Four => "四指",
            GestureKind::ThumbUp => "大拇指向上",
            GestureKind::ThumbDown => "大拇指向下",
            GestureKind::Ok => "OK",
            GestureKind::Pinch => "捏合 / kneading",
            GestureKind::FingerHeart => "比心",
            GestureKind::ILoveYou => "I ❤️ U",
            GestureKind::Rock => "摇滚",
            GestureKind::Unknown => "未知手势",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            GestureKind::OpenPalm => "🖐 ",
            GestureKind::Fist => "✊ ",
            GestureKind::Point => "👉 ",
            GestureKind::Victory => "✌️ ",
            GestureKind::Three => "🤟 ",
            GestureKind::Four => "🖖 ",
            GestureKind::ThumbUp => "👍 ",
            GestureKind::ThumbDown => "👎 ",
            GestureKind::Ok => "👌 ",
            GestureKind::Pinch => "🤏 ",
            GestureKind::FingerHeart => "🫰 ",
            GestureKind::ILoveYou => "🤟 ",
            GestureKind::Rock => "🤘 ",
            GestureKind::Unknown => "⋯ ",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureMotion {
    Steady,
    Fanning,
    VerticalWave,
    Moving,
}

impl GestureMotion {
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            GestureMotion::Steady => "保持",
            GestureMotion::Fanning => "左右扇动",
            GestureMotion::VerticalWave => "上下挥动",
            GestureMotion::Moving => "移动中",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GestureDetail {
    pub primary: GestureKind,
    pub secondary: Option<GestureKind>,
    pub handedness: Handedness,
    pub finger_states: [FingerState; 5],
    pub motion: GestureMotion,
}
