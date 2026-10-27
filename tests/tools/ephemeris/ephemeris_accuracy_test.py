from collections.abc import Callable

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
import rust_ephem
from astropy.constants import R_sun  # type: ignore[import-untyped]
from astropy.coordinates import AltAz, SkyCoord, get_body  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums import SolarSystemObject
from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.base import R_moon


class TestGroundEphemerisApparentPositions:
    """Test suite for the apparent body positions of the rust-ephem based ground ephemeris."""

    @pytest.mark.parametrize(
        "body", ["sun", "moon", "earth", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    )
    def test_ground_body_matches_astropy_apparent_position(
        self,
        keck_ground_ephemeris: Ephemeris,
        body: str,
        body_position_tolerances: dict[str, u.Quantity],
        body_position_factory: Callable[[Ephemeris, str], SkyCoord],
        astropy_apparent_position_factory: Callable[[Ephemeris, str], SkyCoord],
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
    ) -> None:
        """Test that body positions match astropy's apparent positions."""
        separation = max_separation_factory(
            body_position_factory(keck_ground_ephemeris, body),
            astropy_apparent_position_factory(keck_ground_ephemeris, body),
        )
        assert separation < body_position_tolerances[body]

    def test_ground_sun_has_aberration_applied(
        self,
        keck_ground_ephemeris: Ephemeris,
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
        rust_ephem_factory: Callable[[Ephemeris], rust_ephem.Ephemeris],
    ) -> None:
        """Test that the apparent Sun differs from the geometric Sun by about the annual aberration."""
        separation = max_separation_factory(
            keck_ground_ephemeris.sun, rust_ephem_factory(keck_ground_ephemeris).sun
        )
        assert 15 * u.arcsec < separation < 30 * u.arcsec

    def test_ground_sun_distance_matches_astropy(
        self,
        keck_ground_ephemeris: Ephemeris,
        astropy_apparent_position_factory: Callable[[Ephemeris, str], SkyCoord],
    ) -> None:
        """Test that the Sun's distance matches astropy's apparent distance."""
        astropy_sun = astropy_apparent_position_factory(keck_ground_ephemeris, "sun")
        assert u.allclose(keck_ground_ephemeris.sun.distance, astropy_sun.distance, rtol=1e-6)

    def test_ground_sun_radius_angle_matches_distance(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the Sun's angular radius is consistent with its distance."""
        assert u.allclose(
            keck_ground_ephemeris.sun_radius_angle,
            np.arcsin(R_sun / keck_ground_ephemeris.sun.distance),
            atol=0.01 * u.arcsec,
        )

    def test_ground_moon_radius_angle_matches_distance(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the Moon's angular radius is consistent with its distance."""
        assert u.allclose(
            keck_ground_ephemeris.moon_radius_angle,
            np.arcsin(R_moon / keck_ground_ephemeris.moon.distance),
            atol=0.1 * u.arcsec,
        )


class TestTLEEphemerisApparentPositions:
    """Test suite for the apparent body positions of the rust-ephem based TLE ephemeris."""

    @pytest.mark.parametrize(
        "body", ["sun", "moon", "earth", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    )
    def test_tle_body_matches_astropy_apparent_position(
        self,
        hubble_tle_ephemeris: Ephemeris,
        body: str,
        body_position_tolerances: dict[str, u.Quantity],
        body_position_factory: Callable[[Ephemeris, str], SkyCoord],
        astropy_apparent_position_factory: Callable[[Ephemeris, str], SkyCoord],
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
    ) -> None:
        """Test that body positions match astropy's apparent positions."""
        separation = max_separation_factory(
            body_position_factory(hubble_tle_ephemeris, body),
            astropy_apparent_position_factory(hubble_tle_ephemeris, body),
        )
        assert separation < body_position_tolerances[body]

    def test_tle_sun_has_aberration_applied(
        self,
        hubble_tle_ephemeris: Ephemeris,
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
        rust_ephem_factory: Callable[[Ephemeris], rust_ephem.Ephemeris],
    ) -> None:
        """Test that the apparent Sun differs from the geometric Sun by about the annual aberration."""
        separation = max_separation_factory(
            hubble_tle_ephemeris.sun, rust_ephem_factory(hubble_tle_ephemeris).sun
        )
        assert 15 * u.arcsec < separation < 30 * u.arcsec

    def test_tle_spacecraft_velocity_aberration_included(
        self,
        hubble_tle_ephemeris: Ephemeris,
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
    ) -> None:
        """
        Test that the apparent Sun includes aberration due to the spacecraft's
        orbital velocity, so it differs from astropy's apparent Sun computed
        with only the velocity of Earth's rotation.
        """
        astropy_sun = get_body("sun", hubble_tle_ephemeris.timestamp, hubble_tle_ephemeris.earth_location)
        assert max_separation_factory(hubble_tle_ephemeris.sun, astropy_sun) > 2 * u.arcsec

    def test_tle_sun_distance_matches_astropy(
        self,
        hubble_tle_ephemeris: Ephemeris,
        astropy_apparent_position_factory: Callable[[Ephemeris, str], SkyCoord],
    ) -> None:
        """Test that the Sun's distance matches astropy's apparent distance."""
        astropy_sun = astropy_apparent_position_factory(hubble_tle_ephemeris, "sun")
        assert u.allclose(hubble_tle_ephemeris.sun.distance, astropy_sun.distance, rtol=1e-6)

    def test_tle_sun_radius_angle_matches_distance(self, hubble_tle_ephemeris: Ephemeris) -> None:
        """Test that the Sun's angular radius is consistent with its distance."""
        assert u.allclose(
            hubble_tle_ephemeris.sun_radius_angle,
            np.arcsin(R_sun / hubble_tle_ephemeris.sun.distance),
            atol=0.01 * u.arcsec,
        )

    def test_tle_moon_radius_angle_matches_distance(self, hubble_tle_ephemeris: Ephemeris) -> None:
        """Test that the Moon's angular radius is consistent with its distance."""
        assert u.allclose(
            hubble_tle_ephemeris.moon_radius_angle,
            np.arcsin(R_moon / hubble_tle_ephemeris.moon.distance),
            atol=0.1 * u.arcsec,
        )


class TestGetBody:
    """Test suite for the Ephemeris.get_body method."""

    def test_get_body_is_cached(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that repeated calls return the cached body positions."""
        assert keck_ground_ephemeris.get_body("mars") is keck_ground_ephemeris.get_body("mars")

    def test_get_body_accepts_enum(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that SolarSystemObject and string names share the same cache entry."""
        assert keck_ground_ephemeris.get_body(SolarSystemObject.MARS) is keck_ground_ephemeris.get_body(
            "mars"
        )

    def test_get_body_earth_is_earth_attribute(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the earth attribute is the same as get_body('earth')."""
        assert keck_ground_ephemeris.earth is keck_ground_ephemeris.get_body(SolarSystemObject.EARTH)

    def test_get_body_length_matches_timestamp(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that body positions are computed for every timestamp."""
        assert len(keck_ground_ephemeris.get_body("jupiter")) == len(keck_ground_ephemeris.timestamp)

    def test_get_body_astropy_fallback(
        self,
        keck_mock_ephemeris: Ephemeris,
        max_separation_factory: Callable[[SkyCoord, SkyCoord], u.Quantity],
    ) -> None:
        """Test that ephemerides not based on rust-ephem use astropy's get_body."""
        expected = get_body("mars", keck_mock_ephemeris.timestamp, keck_mock_ephemeris.earth_location)
        assert max_separation_factory(keck_mock_ephemeris.get_body("mars"), expected) < 1e-6 * u.arcsec


class TestGroundEphemerisAccuracy:
    """Test suite for the accuracy of the rust-ephem based ground ephemeris against astropy."""

    def test_ground_timestamps_match_rust_ephem(
        self,
        keck_ground_ephemeris: Ephemeris,
        rust_ephem_factory: Callable[[Ephemeris], rust_ephem.Ephemeris],
    ) -> None:
        """Test that the rust-ephem timestamps match the ephemeris timestamps."""
        rust_timestamp = Time(rust_ephem_factory(keck_ground_ephemeris).timestamp)
        assert np.allclose(rust_timestamp.unix, keck_ground_ephemeris.timestamp.unix, atol=1e-3)

    def test_ground_gcrs_matches_astropy(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the observatory GCRS position matches astropy's to within a few tens of meters."""
        astropy_gcrs = keck_ground_ephemeris.earth_location.get_gcrs(keck_ground_ephemeris.timestamp)
        difference = keck_ground_ephemeris.gcrs.cartesian.xyz - astropy_gcrs.cartesian.xyz
        assert np.linalg.norm(difference.to_value(u.m), axis=0).max() < 25

    def test_ground_sun_altitude_matches_astropy(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the Sun altitude matches astropy's."""
        altaz = AltAz(obstime=keck_ground_ephemeris.timestamp, location=keck_ground_ephemeris.earth_location)
        astropy_sun = get_body("sun", keck_ground_ephemeris.timestamp, keck_ground_ephemeris.earth_location)
        assert u.allclose(
            keck_ground_ephemeris.sun.transform_to(altaz).alt,
            astropy_sun.transform_to(altaz).alt,
            atol=0.1 * u.arcsec,
        )

    def test_ground_longitude_east_of_180_matches_west(
        self, east_longitude_ground_ephemeris: Ephemeris, west_longitude_ground_ephemeris: Ephemeris
    ) -> None:
        """Test that a longitude in the range 180-360 degrees matches the equivalent negative longitude."""
        assert u.allclose(
            east_longitude_ground_ephemeris.gcrs.cartesian.xyz,
            west_longitude_ground_ephemeris.gcrs.cartesian.xyz,
            atol=1 * u.mm,
        )
