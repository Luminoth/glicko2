//! Glicko2 implementation
//!
//! https://www.glicko.net/glicko/glicko2.pdf
//! https://en.wikipedia.org/wiki/Glicko_rating_system
//!
//! Volatility measure indicates the degree of expected fluctuation in a player's rating.
//! This is high when a player has erratic performances and low when the player performs at a consistent level.
//!
//! Player's strength summarized as an interval rather than just a rating:
//!     95% confidence interval: [r - 2 * RD..r + 2 * RD]

// TODO: re-do this as a multi-algorithm crate that includes Weng-Lin for team-based games:
//      https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf
//      https://www.csie.ntu.edu.tw/~cjlin/papers/online_ranking/online_journal.pdf

use std::ops::Range;

use thiserror::Error;
use tracing::warn;

#[derive(Error, Debug)]
pub enum Glicko2Error {
    #[error("not enough outcomes to compute updated skills")]
    NotEnoughOutcomes,

    #[error("too many outcomes to compute updated skills")]
    TooManyOutcomes,
}

/// System constant (τ), constrains the change in volatility over time.
///
/// Smaller values prevent the volatility measures from changing by large amounts,
/// which in turn prevent enormous changes in ratings based on very improbable results.
// TODO: this should be configurable, generally between 0.2 and 1.2
pub const TAU: f64 = 0.5;

/// Glicko unrated rating
pub const UNRATED_RATING: f64 = 1500.0;

/// Glicko unrated rating deviation
pub const UNRATED_RATING_DEVIATION: f64 = 350.0;

/// Glicko unrated volatility
pub const UNRATED_VOLATILITY: f64 = 0.06;

/// Glick to Glicko2 conversion scale
pub const GLICKO2_SCALE: f64 = 173.7178;

/// Convergence tolerance
const EPSILON: f64 = 0.000001;

const PI2: f64 = std::f64::consts::PI * std::f64::consts::PI;

/// Convert Glicko2 values to Glicko values
pub fn glicko2_to_glicko(rating: f64, rating_deviation: f64) -> (f64, f64) {
    (
        (GLICKO2_SCALE * rating) + UNRATED_RATING,
        GLICKO2_SCALE * rating_deviation,
    )
}

/// Convert Glicko values to Glicko2 values
pub fn glicko_to_glicko2(rating: f64, rating_deviation: f64) -> (f64, f64) {
    (
        (rating - UNRATED_RATING) / GLICKO2_SCALE,
        rating_deviation / GLICKO2_SCALE,
    )
}

fn g(skill: &PlayerSkill) -> f64 {
    let rd2 = skill.glicko2_rating_deviation().powf(2.0);

    1.0 / (1.0 + (3.0 * rd2 / PI2)).sqrt()
}

#[allow(non_snake_case)]
fn E(player: &PlayerSkill, opponent: &PlayerSkill) -> f64 {
    1.0 / (1.0 + f64::exp(-g(opponent) * (player.glicko2_rating() - opponent.glicko2_rating())))
}

fn f(player: &PlayerSkill, x: f64, v: f64, delta: f64) -> f64 {
    let a = player.volatility().powf(2.0).ln();
    let ex = f64::exp(x);
    let d2 = delta.powf(2.0);
    let rd2 = player.glicko2_rating_deviation().powf(2.0);

    ((ex * (d2 - rd2 - v - ex)) / (2.0 * (rd2 + v + ex).powf(2.0))) - ((x - a) / (TAU * TAU))
}

#[allow(non_snake_case)]
fn updated_volatility(player: &PlayerSkill, v: f64, delta: f64) -> f64 {
    let d2 = delta.powf(2.0);
    let rd2 = player.glicko2_rating_deviation().powf(2.0);

    let A = player.volatility().powf(2.0).ln();
    let mut B = if d2 > (rd2 + v) {
        (d2 - rd2 - v).ln()
    } else {
        // bracket ln(volatility^2)
        // k should almost always be 1, very rarely 2 or more
        let mut k = 1;
        loop {
            if f(player, A - k as f64 * TAU, v, delta) >= 0.0 {
                break;
            }
            k += 1;
        }
        A - k as f64 * TAU
    };
    let mut A = A;

    let mut fA = f(player, A, v, delta);
    let mut fB = f(player, B, v, delta);

    // main Illinois algorithm iteration
    // find A such that f(A) = 0
    let mut iterations = 0;
    loop {
        if (B - A).abs() <= EPSILON {
            break;
        }

        let C = A + (A - B) * fA / (fB - fA);
        let fC = f(player, C, v, delta);
        if fC * fB <= 0.0 {
            A = B;
            fA = fB;
        } else {
            fA /= 2.0;
        }

        B = C;
        fB = fC;

        iterations += 1;
        if iterations == 20 {
            warn!("exceeding max glicko2 iterations");
            // TODO: do we need to adjust anything to make sure the values are legit?
            break;
        }
    }

    f64::exp(A / 2.0)
}

/// Outcome of a single game
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}

impl Outcome {
    /// Gets the Glicko2 score value for the outcome
    #[inline]
    pub fn score(&self) -> f64 {
        match self {
            Self::Loss => 0.0,
            Self::Draw => 0.5,
            Self::Win => 1.0,
        }
    }
}

/// Player skill container
///
/// Values are scaled for Glicko2
#[derive(Debug, Copy, Clone)]
pub struct PlayerSkill {
    rating: f64,
    rating_deviation: f64,
    volatility: f64,
}

impl Default for PlayerSkill {
    fn default() -> Self {
        let (rating, rating_deviation) =
            glicko_to_glicko2(UNRATED_RATING, UNRATED_RATING_DEVIATION);

        Self {
            rating,
            rating_deviation,
            volatility: UNRATED_VOLATILITY,
        }
    }
}

impl PlayerSkill {
    /// Create a new player skill container with the given Glicko values
    #[inline]
    pub fn new_glicko(rating: f64, rating_deviation: f64, volatility: f64) -> Self {
        let (rating, rating_deviation) = glicko_to_glicko2(rating, rating_deviation);

        Self {
            rating,
            rating_deviation,
            volatility,
        }
    }

    /// Create a new player skill container with the given Glicko2 values
    #[inline]
    pub fn new_glicko2(rating: f64, rating_deviation: f64, volatility: f64) -> Self {
        Self {
            rating,
            rating_deviation,
            volatility,
        }
    }

    /// Create a new player skill container with the given Glicko values that is intended to be used as an opponent
    /// (Opponents ignore their volatility value)
    #[inline]
    pub fn new_opponent_glicko(rating: f64, rating_deviation: f64) -> Self {
        let (rating, rating_deviation) = glicko_to_glicko2(rating, rating_deviation);

        Self {
            rating,
            rating_deviation,
            volatility: UNRATED_VOLATILITY,
        }
    }

    /// Create a new player skill container with the given Glicko2 values that is intended to be used as an opponent
    /// (Opponents ignore their volatility value)
    #[inline]
    pub fn new_opponent_glicko2(rating: f64, rating_deviation: f64) -> Self {
        Self {
            rating,
            rating_deviation,
            volatility: UNRATED_VOLATILITY,
        }
    }

    /// Glicko rating (r)
    #[inline]
    pub fn glicko_rating(&self) -> f64 {
        let (rating, _) = glicko2_to_glicko(self.rating, self.rating_deviation);
        rating
    }

    /// Glicko2 rating (µ)
    #[inline]
    pub fn glicko2_rating(&self) -> f64 {
        self.rating
    }

    /// Glicko rating deviation (RD)
    #[inline]
    pub fn glicko_rating_deviation(&self) -> f64 {
        let (_, rating_deviation) = glicko2_to_glicko(self.rating, self.rating_deviation);
        rating_deviation
    }

    /// Glicko2 rating deviation (φ)
    #[inline]
    pub fn glicko2_rating_deviation(&self) -> f64 {
        self.rating_deviation
    }

    /// Glicko / Glicko2 volatility (σ)
    #[inline]
    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    /// 95% confidence player strength interval (2 standard deviations)
    #[inline]
    pub fn strength_interval(&self) -> Range<u64> {
        (self.glicko_rating() as u64 - 2 * self.glicko_rating_deviation() as u64)
            ..(self.glicko_rating().ceil() as u64
                + 2 * self.glicko_rating_deviation().ceil() as u64)
    }

    /// Computes the updated skill values for the player from the provided outcomes
    ///
    /// Ideally want to have 10-15 outcomes for this
    pub fn compute_updated_skill(
        &self,
        outcomes: impl AsRef<[(PlayerSkill, Outcome)]>,
    ) -> PlayerSkill {
        let outcomes = outcomes.as_ref();

        // if the player did not compete during the rating period
        // then we only increase their RD
        if outcomes.is_empty() {
            return PlayerSkill::new_glicko2(
                self.rating,
                self.glicko2_rating_deviation().powf(2.0) + self.volatility().powf(2.0),
                self.volatility,
            );
        }

        let v = 1.0
            / outcomes
                .iter()
                .map(|(opponent, _)| {
                    g(opponent).powf(2.0) * E(self, opponent) * (1.0 - E(self, opponent))
                })
                .sum::<f64>();

        let delta = v * outcomes
            .iter()
            .map(|(opponent, outcome)| g(opponent) * (outcome.score() - E(self, opponent)))
            .sum::<f64>();

        let volatility = updated_volatility(self, v, delta);

        let rating_deviation =
            (self.glicko2_rating_deviation().powf(2.0) + volatility.powf(2.0)).sqrt();
        let rating_deviation = 1.0 / ((1.0 / rating_deviation.powf(2.0)) + (1.0 / v)).sqrt();

        let rating = self.glicko2_rating()
            + (rating_deviation.powf(2.0)
                * outcomes
                    .iter()
                    .map(|(opponent, outcome)| g(opponent) * (outcome.score() - E(self, opponent)))
                    .sum::<f64>());

        PlayerSkill::new_glicko2(rating, rating_deviation, volatility)
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn rating_conversion() {
        let player = PlayerSkill::new_glicko(1500.0, 200.0, 0.06);
        assert_approx_eq!(player.glicko2_rating(), 0.0);

        let opponent = PlayerSkill::new_opponent_glicko(1400.0, 30.0);
        assert_approx_eq!(opponent.glicko2_rating(), -0.5756, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1550.0, 100.0);
        assert_approx_eq!(opponent.glicko2_rating(), 0.2878, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1700.0, 300.0);
        assert_approx_eq!(opponent.glicko2_rating(), 1.1513, 0.0001);
    }

    #[test]
    fn rating_deviation_conversion() {
        let player = PlayerSkill::new_glicko(1500.0, 200.0, 0.06);
        assert_approx_eq!(player.glicko2_rating_deviation(), 1.1513, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1400.0, 30.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 0.1727, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1550.0, 100.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 0.5756, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1700.0, 300.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 1.7269, 0.0001);
    }

    #[test]
    fn opponent_g() {
        let opponent = PlayerSkill::new_opponent_glicko(1400.0, 30.0);
        assert_approx_eq!(g(&opponent), 0.9955, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1550.0, 100.0);
        assert_approx_eq!(g(&opponent), 0.9531, 0.0001);

        let opponent = PlayerSkill::new_opponent_glicko(1700.0, 300.0);
        assert_approx_eq!(g(&opponent), 0.7242, 0.0001);
    }

    #[test]
    #[allow(non_snake_case)]
    fn opponent_E() {
        let player = PlayerSkill::new_glicko(1500.0, 200.0, 0.06);

        let opponent = PlayerSkill::new_opponent_glicko(1400.0, 30.0);
        assert_approx_eq!(E(&player, &opponent), 0.639, 0.001);

        let opponent = PlayerSkill::new_opponent_glicko(1550.0, 100.0);
        assert_approx_eq!(E(&player, &opponent), 0.432, 0.001);

        let opponent = PlayerSkill::new_opponent_glicko(1700.0, 300.0);
        assert_approx_eq!(E(&player, &opponent), 0.303, 0.001);
    }

    #[test]
    fn basic() {
        let player = PlayerSkill::new_glicko(1500.0, 200.0, 0.06);

        let outcomes = vec![
            (PlayerSkill::new_opponent_glicko(1400.0, 30.0), Outcome::Win),
            (
                PlayerSkill::new_opponent_glicko(1550.0, 100.0),
                Outcome::Loss,
            ),
            (
                PlayerSkill::new_opponent_glicko(1700.0, 300.0),
                Outcome::Loss,
            ),
        ];

        let updated_skill = player.compute_updated_skill(outcomes);
        assert_approx_eq!(updated_skill.glicko_rating(), 1464.06, 0.01);
        assert_approx_eq!(updated_skill.glicko_rating_deviation(), 151.52, 0.01);
        assert_approx_eq!(updated_skill.volatility(), 0.05999, 0.0001);
    }

    #[test]
    fn no_outcomes() {
        let player = PlayerSkill::new_glicko(1500.0, 200.0, 0.06);

        let updated_skill = player.compute_updated_skill(vec![]);
        assert_approx_eq!(updated_skill.glicko_rating(), 1500.0, 0.0001);
        assert_approx_eq!(updated_skill.glicko_rating_deviation(), 230.8838, 0.0001);
        assert_approx_eq!(updated_skill.volatility(), 0.06, 0.0001);
    }
}
