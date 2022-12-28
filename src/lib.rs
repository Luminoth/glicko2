//! Glicko2 implementation
//!
//! https://www.glicko.net/glicko/glicko2.pdf
//!
//! Volatility measure indicates the degree of expected fluctuation in a player's rating.
//! This is high when a player has erratic performances and low when the player performs at a consistent level.
//!
//! Player's strength summarized as an interval rather than just a rating:
//!     95% confidence interval: [r - 2 * RD..r + 2 * RD]

use std::ops::Range;

use thiserror::Error;

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
// TOOD: this should be configurable between 0.3 and 1.2
pub const TAU: f64 = 0.3;

const TAU2: f64 = TAU * TAU;

/// Glicko unrated rating
pub const UNRATED_RATING: u64 = 1500;

/// Glicko unrated rating deviation
pub const UNRATED_RATING_DEVIATION: u64 = 350;

/// Glicko unrated volatility
pub const UNRATED_VOLATILITY: f64 = 0.06;

/// Glick to Glicko2 conversion ratio
pub const GLICKO2_RATIO: f64 = 173.7178;

/// Convergence tolerance
const EPSILON: f64 = 0.000001;

const PI2: f64 = std::f64::consts::PI * std::f64::consts::PI;

fn g(skill: &PlayerSkill) -> f64 {
    let rd2 = skill.glicko2_rating_deviation().powf(2.0);

    1.0 / (1.0 + (3.0 * rd2 / PI2)).sqrt()
}

#[allow(non_snake_case)]
fn E(player: &PlayerSkill, opponent: &PlayerSkill) -> f64 {
    1.0 / (1.0
        + 10.0_f64.powf(-g(opponent) * (player.glicko2_rating() - opponent.glicko2_rating())))
}

fn f(player: &PlayerSkill, x: f64, v: f64, delta: f64) -> f64 {
    let a = player.volatility().powf(2.0).ln();
    let ex = std::f64::consts::E.powf(x);
    let d2 = delta.powf(2.0);
    let rd2 = player.glicko2_rating_deviation().powf(2.0);

    ((ex * (d2 - rd2 - v - ex)) / (2.0 * (rd2 + v + ex).powf(2.0))) - ((x - a) / TAU2)
}

#[allow(non_snake_case)]
fn updated_volatility(player: &PlayerSkill, v: f64, delta: f64) -> f64 {
    let d2 = delta.powf(2.0);
    let rd2 = player.glicko2_rating_deviation().powf(2.0);

    // TODO: this special case is not handled
    //assert!(d2 > rd2 + v, "{} <= {} + {} ({})", d2, rd2, v, rd2 + v);

    let A = player.volatility().powf(2.0).ln();
    let mut B = if d2 > (rd2 + v) {
        (d2 - rd2 - v).ln()
    } else {
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

        // TODO: this should be handled better
        assert!(iterations < 20);
    }

    std::f64::consts::E.powf(A / 2.0)
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
#[derive(Debug, Copy, Clone)]
pub struct PlayerSkill {
    rating: f64,
    rating_deviation: f64,
    volatility: f64,
}

impl Default for PlayerSkill {
    fn default() -> Self {
        Self {
            rating: UNRATED_RATING as f64,
            rating_deviation: UNRATED_RATING_DEVIATION as f64,
            volatility: UNRATED_VOLATILITY,
        }
    }
}

impl PlayerSkill {
    /// Create a new player skill container with the given values
    #[inline]
    pub fn new(rating: f64, rating_deviation: f64, volatility: f64) -> Self {
        Self {
            rating,
            rating_deviation,
            volatility,
        }
    }

    /// Create a new player skill container with the given values that is intended to be used as an opponent
    /// (Opponents ignore their volatility value)
    #[inline]
    pub fn new_opponent(rating: f64, rating_deviation: f64) -> Self {
        Self {
            rating,
            rating_deviation,
            volatility: UNRATED_VOLATILITY,
        }
    }

    /// Glicko rating (r)
    #[inline]
    pub fn rating(&self) -> f64 {
        self.rating
    }

    /// Glicko2 rating (µ)
    #[inline]
    pub fn glicko2_rating(&self) -> f64 {
        (self.rating - UNRATED_RATING as f64) / GLICKO2_RATIO
    }

    /// Glicko rating deviation (RD)
    #[inline]
    pub fn rating_deviation(&self) -> f64 {
        self.rating_deviation
    }

    /// Glicko2 rating deviation (φ)
    #[inline]
    pub fn glicko2_rating_deviation(&self) -> f64 {
        self.rating_deviation / GLICKO2_RATIO
    }

    /// Glicko / Glicko2 volatility (σ)
    #[inline]
    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    /// 95% confidence player strength interval
    #[inline]
    pub fn strength_interval(&self) -> Range<u64> {
        let r = self.rating() as u64;
        let rd = self.rating_deviation() as u64;

        (r - 2 * rd)..(r + 2 * rd)
    }

    /// Computes the updated skill values for the player from the provided outcomes
    ///
    /// This requires 10-15 outcomes to run successfully
    pub fn compute_updated_skill(
        &self,
        outcomes: impl AsRef<[(PlayerSkill, Outcome)]>,
    ) -> Result<PlayerSkill, Glicko2Error> {
        let outcomes = outcomes.as_ref();

        if outcomes.len() < 10 {
            return Err(Glicko2Error::NotEnoughOutcomes);
        }

        if outcomes.len() > 15 {
            return Err(Glicko2Error::TooManyOutcomes);
        }

        Ok(self.compute_updated_skill_unchecked(outcomes))
    }

    /// Computes the updated skill values for the player from the provided outcomes
    ///
    /// This does not check the number of outcomes
    pub fn compute_updated_skill_unchecked(
        &self,
        outcomes: impl AsRef<[(PlayerSkill, Outcome)]>,
    ) -> PlayerSkill {
        let outcomes = outcomes.as_ref();

        // if the player did not compete during the rating period
        // then we only increase their RD
        if outcomes.is_empty() {
            return PlayerSkill {
                rating: self.rating,
                rating_deviation: self.glicko2_rating_deviation().powf(2.0)
                    + self.volatility().powf(2.0),
                volatility: self.volatility,
            };
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

        PlayerSkill {
            rating: (GLICKO2_RATIO * rating) + UNRATED_RATING as f64,
            rating_deviation: GLICKO2_RATIO * rating_deviation,
            volatility,
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn rating_conversion() {
        let player = PlayerSkill::new(1500.0, 200.0, 0.06);
        assert_approx_eq!(player.glicko2_rating(), 0.0);

        let opponent = PlayerSkill::new_opponent(1400.0, 30.0);
        assert_approx_eq!(opponent.glicko2_rating(), -0.5756, 0.0001);

        let opponent = PlayerSkill::new_opponent(1550.0, 100.0);
        assert_approx_eq!(opponent.glicko2_rating(), 0.2878, 0.0001);

        let opponent = PlayerSkill::new_opponent(1700.0, 300.0);
        assert_approx_eq!(opponent.glicko2_rating(), 1.1513, 0.0001);
    }

    #[test]
    fn rating_deviation_conversion() {
        let player = PlayerSkill::new(1500.0, 200.0, 0.06);
        assert_approx_eq!(player.glicko2_rating_deviation(), 1.1513, 0.0001);

        let opponent = PlayerSkill::new_opponent(1400.0, 30.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 0.1727, 0.0001);

        let opponent = PlayerSkill::new_opponent(1550.0, 100.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 0.5756, 0.0001);

        let opponent = PlayerSkill::new_opponent(1700.0, 300.0);
        assert_approx_eq!(opponent.glicko2_rating_deviation(), 1.7269, 0.0001);
    }

    #[test]
    fn opponent_g() {
        let opponent = PlayerSkill::new_opponent(1400.0, 30.0);
        assert_approx_eq!(g(&opponent), 0.9955, 0.0001);

        let opponent = PlayerSkill::new_opponent(1550.0, 100.0);
        assert_approx_eq!(g(&opponent), 0.9531, 0.0001);

        let opponent = PlayerSkill::new_opponent(1700.0, 300.0);
        assert_approx_eq!(g(&opponent), 1.7242, 0.0001);
    }

    #[test]
    #[allow(non_snake_case)]
    fn opponent_E() {
        let player = PlayerSkill::new(1500.0, 200.0, 0.06);

        let opponent = PlayerSkill::new_opponent(1400.0, 30.0);
        assert_approx_eq!(E(&player, &opponent), 0.639, 0.0001);

        let opponent = PlayerSkill::new_opponent(1550.0, 100.0);
        assert_approx_eq!(E(&player, &opponent), 0.432, 0.0001);

        let opponent = PlayerSkill::new_opponent(1700.0, 300.0);
        assert_approx_eq!(E(&player, &opponent), 0.303, 0.0001);
    }

    #[test]
    fn basic() {
        let player = PlayerSkill::new(1500.0, 200.0, 0.06);

        let outcomes = vec![
            (PlayerSkill::new_opponent(1400.0, 30.0), Outcome::Win),
            (PlayerSkill::new_opponent(1550.0, 100.0), Outcome::Loss),
            (PlayerSkill::new_opponent(1700.0, 300.0), Outcome::Loss),
        ];

        let updated_skill = player.compute_updated_skill_unchecked(outcomes);
        assert_approx_eq!(updated_skill.rating(), 1464.06, 0.0001);
        assert_approx_eq!(updated_skill.rating_deviation(), 151.52, 0.0001);
        assert_approx_eq!(updated_skill.volatility(), 0.05999, 0.0001);
    }
}
