//! Library for using exponential moving averages that is generic over the underlying float type.
#![cfg_attr(
  not(test),
  deny(warnings, clippy::all, clippy::pedantic, clippy::cargo, missing_docs, missing_crate_level_docs)
)]
#![deny(unsafe_code)]
#![cfg_attr(not(test), no_std)]

use core::cmp::Ordering;
use core::convert::TryInto;
use core::time::Duration;
use num_traits::identities::{One, Zero};
use num_traits::Float;
use ordered_float::{FloatIsNan, NotNan};

/// A struct representing an exponential moving average
///
/// The weighting can be chosen for each accumulation. To have the weighting be part of the struct see [`StableEma`]
#[must_use]
#[derive(Clone)]
pub struct Ema<F>
where
  F: Float,
{
  mean: NotNan<F>,
  variance: NotNan<F>,
}

impl<F> PartialEq for Ema<F>
where
  F: Float,
{
  fn eq(&self, other: &Self) -> bool {
    self.mean.eq(&other.mean) && self.variance.eq(&other.variance)
  }
}

impl<F> Eq for Ema<F> where F: Float {}

impl<F> PartialOrd for Ema<F>
where
  F: Float,
{
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<F> Ord for Ema<F>
where
  F: Float,
{
  fn cmp(&self, other: &Self) -> Ordering {
    self.mean.cmp(&other.mean).then_with(|| self.variance.cmp(&other.variance))
  }
}

impl<F> Ema<F>
where
  F: Float + TryInto<NotNan<F>>,
{
  /// Tries to create a new `Ema` struct from raw float values
  /// # Errors
  /// Fails if `mean` or `variance` are NaN
  pub fn try_new(
    mean: impl TryInto<NotNan<F>, Error = FloatIsNan>,
    variance: impl TryInto<NotNan<F>, Error = FloatIsNan>,
  ) -> Result<Self, FloatIsNan> {
    Ok(Self::new(mean.try_into()?, variance.try_into()?))
  }
}

impl<F> Ema<F>
where
  F: Float,
{
  /// Returns a new `Ema` struct with the mean and variance estimates already initialized.
  ///
  /// It is recommended to choose these values to be as close to expected as possible so that they can converge quickly
  pub fn new(mean: NotNan<F>, variance: NotNan<F>) -> Self {
    Self { mean, variance }
  }
  /// Accumulates a new value into this `Ema`. The mean and variance are adjusted by the `recent_weight`
  pub fn accumulate(&mut self, value: NotNan<F>, recent_weight: NotNan<F>) {
    let recent_weight = recent_weight.min(NotNan::one()).max(NotNan::zero());
    let mean = self.mean;
    let delta = value - mean;
    let new_mean = mean + recent_weight * delta;
    let new_variance = (NotNan::one() - recent_weight) * (self.variance + recent_weight * delta * delta);
    self.mean = new_mean;
    self.variance = new_variance;
  }
  /// Tries to accumulate raw float values.
  /// # Errors
  /// Fails if `value` or `recent_weight` are NaN
  pub fn try_accumulate(&mut self, value: F, recent_weight: F) -> Result<(), FloatIsNan> {
    let value = NotNan::new(value)?;
    let recent_weight = NotNan::new(recent_weight)?;
    self.accumulate(value, recent_weight);
    Ok(())
  }
  /// Returns the mean of this `Ema`
  #[must_use]
  #[inline]
  pub fn mean(&self) -> NotNan<F> {
    self.mean
  }
  /// Returns the variance of this `Ema`
  #[must_use]
  #[inline]
  pub fn variance(&self) -> NotNan<F> {
    self.variance
  }
  /// Returns the standard deviation of this `Ema`
  #[allow(clippy::missing_panics_doc)]
  #[must_use]
  #[inline]
  pub fn std_dev(&self) -> NotNan<F> {
    // Not using `unwrap` or `expect` because we don't want to force the associated type to be `Debug`
    NotNan::new(self.variance.sqrt()).unwrap_or_else(|_| panic!("sqrt won't return NaN if it didn't start with it"))
  }
  /// Returns the mean of this `Ema` as a duration in seconds. Useful when using an `Ema` to time events.
  #[must_use]
  #[inline]
  pub fn mean_duration(&self) -> Duration {
    Duration::from_secs_f64(self.mean().to_f64().unwrap_or(0.0).max(0.0))
  }
  /// Returns the standard deviation of this `Ema` as a duration in seconds. Useful when using an `Ema` to time events
  #[must_use]
  #[inline]
  pub fn std_dev_duration(&self) -> Duration {
    Duration::from_secs_f64(self.std_dev().to_f64().unwrap_or(0.0).max(0.0))
  }
}

impl<F> Default for Ema<F>
where
  F: Float,
{
  fn default() -> Self {
    Self {
      mean: NotNan::zero(),
      variance: NotNan::zero(),
    }
  }
}

impl<F> core::fmt::Debug for Ema<F>
where
  F: Float + core::fmt::Debug,
{
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("EMA")
      .field("mean", &*self.mean)
      .field("variance", &*self.variance)
      .finish()
  }
}

/// A stable [Ema] where the `recent_weight` is set at initialization and the same value is always used.
#[derive(Clone)]
#[must_use]
pub struct StableEma<F>
where
  F: Float,
{
  ema: Ema<F>,
  recent_weight: NotNan<F>,
}

impl<F> PartialEq for StableEma<F>
where
  F: Float,
{
  fn eq(&self, other: &Self) -> bool {
    self.ema.eq(&other.ema) && self.recent_weight.eq(&other.recent_weight)
  }
}

impl<F> Eq for StableEma<F> where F: Float {}

impl<F> PartialOrd for StableEma<F>
where
  F: Float,
{
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<F> Ord for StableEma<F>
where
  F: Float,
{
  fn cmp(&self, other: &Self) -> Ordering {
    self.ema.cmp(&other.ema).then_with(|| self.recent_weight.cmp(&other.recent_weight))
  }
}

impl<F> Default for StableEma<F>
where
  F: Float,
{
  fn default() -> Self {
    Self {
      ema: Ema::default(),
      // Doing panics and stuff to avoid trait bounds.
      recent_weight: NotNan::new(F::from(0.1).unwrap_or_else(|| panic!("cannot fail"))).unwrap_or_else(|_| panic!("inner is a number")),
    }
  }
}

impl<F> StableEma<F>
where
  F: Float,
{
  /// Returns a new `StableEma` with the `mean`, `variance`, and `recent_weight` all initialized.
  ///
  /// It is recommended to choose the `mean` and `variance` to be as close to expected as possible so that they can converge quickly
  pub fn new(mean: NotNan<F>, variance: NotNan<F>, recent_weight: NotNan<F>) -> Self {
    Self {
      ema: Ema::new(mean, variance),
      recent_weight,
    }
  }

  /// Tries to create a new `StableEma` from raw float values.
  /// # Errors
  /// Fails if `mean`, `variance`, or `recent_weight` are NaN
  pub fn try_new<T: TryInto<NotNan<F>, Error = FloatIsNan>>(mean: T, variance: T, recent_weight: T) -> Result<Self, FloatIsNan> {
    Ok(Self::new(mean.try_into()?, variance.try_into()?, recent_weight.try_into()?))
  }

  /// Accumulates the value to this `StableEma`
  pub fn accumulate(&mut self, value: NotNan<F>) {
    self.ema.accumulate(value, self.recent_weight)
  }

  /// Tries to accumulate a raw float value
  /// # Errors
  /// Fails if `value` is NaN
  pub fn try_accumulate(&mut self, value: F) -> Result<(), FloatIsNan> {
    self.accumulate(NotNan::new(value)?);
    Ok(())
  }

  /// Returns the mean of this `StableEma`
  #[inline]
  #[must_use]
  pub fn mean(&self) -> NotNan<F> {
    self.ema.mean()
  }

  /// Returns the variance of this `StableEma`
  #[inline]
  #[must_use]
  pub fn variance(&self) -> NotNan<F> {
    self.ema.variance()
  }

  /// Returns the standard deviation of this `StableEma`
  #[inline]
  #[must_use]
  pub fn std_dev(&self) -> NotNan<F> {
    self.ema.std_dev()
  }

  /// Returns the recent weight that this `StableEma` uses to accumulate values
  #[must_use]
  pub fn recent_weight(&self) -> NotNan<F> {
    self.recent_weight
  }

  /// Returns the mean of this `StableEma` as a duration in seconds. Useful when using an `Ema` to time events.
  #[inline]
  #[must_use]
  pub fn mean_duration(&self) -> Duration {
    self.ema.mean_duration()
  }

  /// Returns the standard deviation of this `StableEma` as a duration in seconds. Useful when using an `Ema` to time events.
  #[inline]
  #[must_use]
  pub fn std_dev_duration(&self) -> Duration {
    self.ema.std_dev_duration()
  }

  /// Change the recent weight.
  /// # Safety
  /// This is not unsafe to call, but it violates the notion that this has
  /// a stable recent weight
  #[allow(unsafe_code)]
  pub unsafe fn set_recent_weight(&mut self, recent_weight: NotNan<F>) {
    self.recent_weight = recent_weight;
  }
}

impl<F> core::fmt::Debug for StableEma<F>
where
  F: Float + core::fmt::Debug,
{
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("StableEMA")
      .field("mean", &*self.ema.mean)
      .field("variance", &*self.ema.variance)
      .field("recent_weight", &*self.recent_weight)
      .finish()
  }
}

#[cfg(test)]
mod test {
  use super::*;

  fn test_ema<F: Float + num_traits::FromPrimitive + core::fmt::Debug>(iters: u32, mean_epsilon: F, variance_epsilon: F) {
    let mut ema = StableEma::<F>::new(NotNan::one(), NotNan::zero(), NotNan::new(F::from_f64(0.2).unwrap()).unwrap());
    assert_eq!(*ema.mean(), F::one());
    assert_eq!(*ema.variance(), F::zero());
    assert_eq!(*ema.std_dev(), F::zero());
    assert_eq!(ema.mean_duration(), Duration::from_secs(1));
    assert_eq!(ema.std_dev_duration(), Duration::from_secs(0));
    assert_eq!(*ema.recent_weight(), F::from_f64(0.2).unwrap());
    (0..10000).for_each(|_| ema.accumulate(NotNan::one()));
    assert_eq!(ema.mean(), NotNan::one());
    assert_eq!(ema.variance(), NotNan::zero());

    (1..=iters).for_each(|i| {
      ema.accumulate(NotNan::new(F::from(i as f64).unwrap()).unwrap());
      if i > iters / 2 {
        assert!(
          (ema.mean() - F::from((i - 4) as f64).unwrap()).abs() <= mean_epsilon,
          "mean: {:?}",
          ema.mean()
        );
        assert!(
          (ema.variance() - F::from(20.0).unwrap()).abs() <= variance_epsilon,
          "variance: {:?}",
          ema.variance()
        );
        assert!(
          (ema.std_dev() - F::from(20.0.sqrt()).unwrap()).abs() <= variance_epsilon,
          "std_dev: {:?}",
          ema.std_dev()
        );
      }
    });
  }

  #[test]
  fn test_types() {
    use half::{bf16, f16};
    test_ema::<f32>(10000, 1e-7, 1e-5);
    let mut ema = StableEma::<f32>::default();
    ema.try_accumulate(f32::NAN).unwrap_err();
    test_ema::<f64>(100000, 1e-7, 1e-5);
    let mut ema = StableEma::<f64>::default();
    ema.try_accumulate(f64::NAN).unwrap_err();
    test_ema::<bf16>(250, bf16::from_f32(1e-7), bf16::from_f32(0.25));
    let mut ema = Ema::<bf16>::default();
    ema.try_accumulate(bf16::from_f32(f32::NAN), bf16::from_f32(0.5)).unwrap_err();
    test_ema::<f16>(500, f16::from_f32(1e-7), f16::from_f32(0.125));
    let mut ema = Ema::<f16>::default();
    ema.try_accumulate(f16::from_f32(f32::NAN), f16::from_f32(0.5)).unwrap_err();
  }
}
