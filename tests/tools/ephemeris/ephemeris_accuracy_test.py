import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.constants import R_sun  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    AltAz,
    CartesianRepresentation,
    EarthLocation,
    Longitude,
    SkyCoord,
    get_body,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from across.tools.core.enums import SolarSystemObject
from across.tools.ephemeris import Ephemeris, compute_ground_ephemeris
from across.tools.ephemeris.base import R_moon

# Maximum allowed difference between the ephemeris body positions and
# astropy's apparent positions. astropy's default "builtin" planetary
# ephemeris is approximate (e.g. ~20 arcsec for Mars and Saturn), whereas
# rust-ephem uses JPL DE440s, so the tolerances for the planets reflect the
# accuracy of astropy's ephemeris, not that of rust-ephem. With astropy also
# using DE440s, all bodies agree to better than 0.01 arcsec.
BODY_TOLERANCES = {
    "sun": 0.1 * u.arcsec,
    "earth": 0.1 * u.arcsec,
    "moon": 2 * u.arcsec,
    "mercury": 1 * u.arcsec,
    "venus": 3 * u.arcsec,
    "mars": 25 * u.arcsec,
    "jupiter": 3 * u.arcsec,
    "saturn": 30 * u.arcsec,
    "uranus": 10 * u.arcsec,
    "neptune": 5 * u.arcsec,
}


def body_position(ephemeris: Ephemeris, body: str) -> SkyCoord:
    """Return the position of a body from the ephemeris attributes, or get_body for planets."""
    if body in ("sun", "moon", "earth"):
        return getattr(ephemeris, body)
    return ephemeris.get_body(body)


def astropy_apparent_position(ephemeris: Ephemeris, body: str) -> SkyCoord:
    """
    Return astropy's apparent position of a body as seen from the observatory,
    including the aberration due to the observatory's full GCRS velocity.
    Passing an EarthLocation to get_body only accounts for the velocity due to
    Earth's rotation, which is not correct for a spacecraft.
    """
    frame = GCRS(
        obstime=ephemeris.timestamp,
        obsgeoloc=ephemeris.gcrs.cartesian.without_differentials(),
        obsgeovel=CartesianRepresentation(ephemeris.gcrs.velocity.d_xyz),
    )
    return get_body(body, ephemeris.timestamp, ephemeris.earth_location).transform_to(frame)


def max_separation(coord1: SkyCoord, coord2: SkyCoord) -> u.Quantity:
    """Return the maximum angular separation between two arrays of coordinates."""
    return SkyCoord(coord1.ra, coord1.dec).separation(SkyCoord(coord2.ra, coord2.dec)).max()


@pytest.fixture(params=["keck_ground_ephemeris", "hubble_tle_ephemeris"])
def rust_ephemeris(request: pytest.FixtureRequest) -> Ephemeris:
    """Fixture providing each of the rust-ephem based ephemeris types."""
    ephemeris: Ephemeris = request.getfixturevalue(request.param)
    return ephemeris


class TestApparentBodyPositions:
    """Test that body positions from rust-ephem based ephemerides are apparent positions."""

    @pytest.mark.parametrize("body", list(BODY_TOLERANCES))
    def test_body_matches_astropy_apparent_position(self, rust_ephemeris: Ephemeris, body: str) -> None:
        """Body positions should match astropy's apparent positions."""
        separation = max_separation(
            body_position(rust_ephemeris, body), astropy_apparent_position(rust_ephemeris, body)
        )
        assert separation < BODY_TOLERANCES[body]

    def test_sun_has_aberration_applied(self, rust_ephemeris: Ephemeris) -> None:
        """
        The apparent Sun should differ from rust-ephem's geometric Sun by
        approximately the annual aberration (~20.5 arcsec).
        """
        assert rust_ephemeris._rust_ephem is not None
        separation = max_separation(rust_ephemeris.sun, rust_ephemeris._rust_ephem.sun)
        assert 15 * u.arcsec < separation < 30 * u.arcsec

    def test_spacecraft_velocity_aberration_included(self, hubble_tle_ephemeris: Ephemeris) -> None:
        """
        For a spacecraft, the apparent Sun should include aberration due to the
        spacecraft's orbital velocity, so it should differ from astropy's
        apparent Sun computed with only the velocity of Earth's rotation.
        """
        astropy_sun = get_body("sun", hubble_tle_ephemeris.timestamp, hubble_tle_ephemeris.earth_location)
        assert max_separation(hubble_tle_ephemeris.sun, astropy_sun) > 2 * u.arcsec

    def test_body_distance_matches_astropy(self, rust_ephemeris: Ephemeris) -> None:
        """The Sun's distance should match astropy's apparent distance."""
        astropy_sun = astropy_apparent_position(rust_ephemeris, "sun")
        assert u.allclose(rust_ephemeris.sun.distance, astropy_sun.distance, rtol=1e-6)

    def test_sun_radius_angle_matches_distance(self, rust_ephemeris: Ephemeris) -> None:
        """The Sun's angular radius should be consistent with its distance."""
        assert u.allclose(
            rust_ephemeris.sun_radius_angle,
            np.arcsin(R_sun / rust_ephemeris.sun.distance),
            atol=0.01 * u.arcsec,
        )

    def test_moon_radius_angle_matches_distance(self, rust_ephemeris: Ephemeris) -> None:
        """The Moon's angular radius should be consistent with its distance."""
        assert u.allclose(
            rust_ephemeris.moon_radius_angle,
            np.arcsin(R_moon / rust_ephemeris.moon.distance),
            atol=0.1 * u.arcsec,
        )


class TestGetBody:
    """Test the Ephemeris.get_body method."""

    def test_get_body_is_cached(self, rust_ephemeris: Ephemeris) -> None:
        """Repeated calls should return the cached body positions."""
        assert rust_ephemeris.get_body("mars") is rust_ephemeris.get_body("mars")

    def test_get_body_accepts_enum(self, rust_ephemeris: Ephemeris) -> None:
        """SolarSystemObject and string names should share the same cache entry."""
        assert rust_ephemeris.get_body(SolarSystemObject.MARS) is rust_ephemeris.get_body("mars")

    def test_earth_attribute_is_get_body_earth(self, rust_ephemeris: Ephemeris) -> None:
        """The earth attribute should be the same as get_body('earth')."""
        assert rust_ephemeris.earth is rust_ephemeris.get_body(SolarSystemObject.EARTH)

    def test_get_body_length_matches_timestamp(self, rust_ephemeris: Ephemeris) -> None:
        """Body positions should be computed for every timestamp."""
        assert len(rust_ephemeris.get_body("jupiter")) == len(rust_ephemeris.timestamp)

    def test_get_body_astropy_fallback(
        self, mock_ephemeris: Ephemeris, keck_latitude: u.Quantity, keck_longitude: u.Quantity
    ) -> None:
        """Ephemerides not based on rust-ephem should use astropy's get_body."""
        mock_ephemeris.earth_location = EarthLocation.from_geodetic(
            lat=keck_latitude, lon=keck_longitude, height=4160 * u.m
        )
        expected = get_body("mars", mock_ephemeris.timestamp, mock_ephemeris.earth_location)
        assert max_separation(mock_ephemeris.get_body("mars"), expected) < 1e-6 * u.arcsec


class TestGroundEphemerisAccuracy:
    """Test the accuracy of the rust-ephem based ground ephemeris against astropy."""

    def test_timestamps_match_rust_ephem(self, keck_ground_ephemeris: Ephemeris) -> None:
        """The rust-ephem timestamps should match the ephemeris timestamps."""
        assert keck_ground_ephemeris._rust_ephem is not None
        rust_timestamp = Time(keck_ground_ephemeris._rust_ephem.timestamp)
        assert np.allclose(rust_timestamp.unix, keck_ground_ephemeris.timestamp.unix, atol=1e-3)

    def test_gcrs_matches_astropy(self, keck_ground_ephemeris: Ephemeris) -> None:
        """The observatory GCRS position should match astropy's to within a few tens of meters."""
        astropy_gcrs = keck_ground_ephemeris.earth_location.get_gcrs(keck_ground_ephemeris.timestamp)
        difference = keck_ground_ephemeris.gcrs.cartesian.xyz - astropy_gcrs.cartesian.xyz
        assert np.linalg.norm(difference.to_value(u.m), axis=0).max() < 25

    def test_sun_altitude_matches_astropy(self, keck_ground_ephemeris: Ephemeris) -> None:
        """The Sun altitude should match astropy's."""
        altaz = AltAz(obstime=keck_ground_ephemeris.timestamp, location=keck_ground_ephemeris.earth_location)
        astropy_sun = get_body("sun", keck_ground_ephemeris.timestamp, keck_ground_ephemeris.earth_location)
        assert u.allclose(
            keck_ground_ephemeris.sun.transform_to(altaz).alt,
            astropy_sun.transform_to(altaz).alt,
            atol=0.1 * u.arcsec,
        )

    def test_longitude_east_of_180_matches_west(
        self,
        ephemeris_begin: Time,
        ephemeris_end: Time,
        ephemeris_step_size: TimeDelta,
        keck_latitude: u.Quantity,
        keck_height: u.Quantity,
    ) -> None:
        """
        A longitude given in the range 180-360 degrees should give the same
        result as the equivalent negative longitude.
        """
        east = compute_ground_ephemeris(
            ephemeris_begin,
            ephemeris_end,
            ephemeris_step_size,
            keck_latitude,
            Longitude(241.8 * u.deg),
            keck_height,
        )
        west = compute_ground_ephemeris(
            ephemeris_begin, ephemeris_end, ephemeris_step_size, keck_latitude, -118.2 * u.deg, keck_height
        )
        assert u.allclose(east.gcrs.cartesian.xyz, west.gcrs.cartesian.xyz, atol=1 * u.mm)
