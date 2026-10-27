from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import rust_ephem
from astropy.constants import R_earth, R_sun  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    Angle,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    get_body,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from ..core.enums import SolarSystemObject

# Define the radii of the Moon (as astropy doesn't)
R_moon = 1737.4 * u.km


def _rust_ephem_body_name(body_name: SolarSystemObject | str) -> str:
    """
    Return the name to use when requesting a Solar System body from rust-ephem.

    The planetary ephemeris kernel shipped with rust-ephem (de440s) only
    contains records for the planetary barycenters, so planets are requested as
    e.g. "mars barycenter". The Earth is the exception: "earth barycenter" is
    the Earth-Moon barycenter, which sits ~4700 km from the geocenter, an error
    of tens of degrees as seen from low Earth orbit, so the geocenter is
    requested instead.

    Parameters
    ----------
    body_name : SolarSystemObject | str
        Name of the body ('earth', 'mars', etc.)

    Returns
    -------
    str
        The body name to pass to `rust_ephem.Ephemeris.get_body`.
    """
    name = body_name.value if isinstance(body_name, SolarSystemObject) else body_name
    if name == SolarSystemObject.EARTH:
        return name
    return f"{name} barycenter"


class Ephemeris(ABC):
    """
    A base class for calculating ephemerides of Astronomical Observatories both
    ground and space-based. This abstract class provides the core functionality
    for computing positions and angular sizes of celestial bodies (Sun, Moon,
    Earth) relative to a spacecraft or observation point. It handles time
    series calculations with specified time intervals and step sizes.

    Parameters
    ----------
    begin : datetime or Time
        Start time of ephemeris calculation
    end : datetime or Time
        End time of ephemeris calculation
    step_size : int, TimeDelta, or timedelta, optional
        Time step between calculations in seconds, by default 60

    Attributes
    ----------
    begin : Time
        Start time of ephemeris calculation
    end : Time
        End time of ephemeris calculation
    step_size : TimeDelta
        Step size of ephemeris calculation
    timestamp : Time
        Array of calculation timestamps
    gcrs : SkyCoord
        Spacecraft position in Geocentric Celestial Reference System (GCRS)
        coordinates
    earth_location : EarthLocation
        Spacecraft position relative to Earth
    moon : SkyCoord
        Moon position relative to spacecraft
    sun : SkyCoord
        Sun position relative to spacecraft
    earth : SkyCoord
        Earth position relative to spacecraft
    longitude : Longitude, optional
        Spacecraft longitude
    latitude : Latitude, optional
        Spacecraft latitude
    height : Quantity, optional
        Spacecraft height above Earth's surface
    earth_radius_angle : Angle
        Angular radius of Earth as seen from spacecraft
    moon_radius_angle : Angle
        Angular radius of Moon as seen from spacecraft
    sun_radius_angle : Angle
        Angular radius of Sun as seen from spacecraft
    distance : Quantity
        Distance from spacecraft to Earth center

    Notes
    -----
    This is an abstract base class that must be subclassed. Subclasses must implement
    the prepare_data() method to set up the spacecraft position before ephemeris
    calculations can be performed.
    """

    # Parameters
    begin: Time  # Start time of ephemeris calculation
    end: Time  # End time of ephemeris calculation
    step_size: TimeDelta  # Step size of ephemeris calculation in seconds

    # Computed values
    timestamp: Time
    gcrs: SkyCoord
    earth_location: EarthLocation
    moon: SkyCoord
    sun: SkyCoord
    earth: SkyCoord
    longitude: Longitude | None = None
    latitude: Latitude | None = None
    height: u.Quantity | None = None
    earth_radius_angle: Angle
    moon_radius_angle: Angle
    sun_radius_angle: Angle
    distance: u.Quantity
    _start_unix: float
    _step_seconds: float
    # rust-ephem ephemeris, set by subclasses that compute their data with rust-ephem
    _rust_ephem: rust_ephem.Ephemeris | None = None
    _body_cache: dict[str, SkyCoord]

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
    ) -> None:
        # Convert begin and end to astropy Time
        self.begin = begin if isinstance(begin, Time) else Time(begin)
        self.end = end if isinstance(end, Time) else Time(end)

        # Convert step_size to TimeDelta
        if isinstance(step_size, TimeDelta):
            self.step_size = step_size
        elif isinstance(step_size, timedelta):
            self.step_size = TimeDelta(step_size)
        elif isinstance(step_size, (int, float)):
            self.step_size = TimeDelta(step_size * u.s)

        # Cache scalars used frequently to avoid repeated unit conversions
        self._step_seconds = float(self.step_size.to_value(u.s))

        # Align begin/end to step grid (floor)
        s = self._step_seconds
        self.begin = Time((self.begin.unix // s) * s, format="unix")
        self.end = Time((self.end.unix // s) * s, format="unix")

        # Cache for index calculations to avoid repeated array access and conversions
        self._start_unix = float(self.begin.unix)

        # Compute range of timestamps
        self.timestamp = self._compute_timestamp()

        # Cache of Solar System body positions, keyed by body name
        self._body_cache = {}

    def __len__(self) -> int:
        return len(self.timestamp)

    def index(self, t: Time) -> int:
        """
        For a given time, return an index for the nearest time in the
        ephemeris. Note that internally converting from Time to unix makes
        this run way faster.

        Parameters
        ----------
        t : Time
            The time to find the nearest index for.

        Returns
        -------
        int
            The index of the nearest time in the ephemeris.
        """
        index = int(np.round((t.unix - self._start_unix) // self._step_seconds))
        assert index >= 0 and index < len(self), "Time outside of ephemeris of range"
        return index

    def _compute_timestamp(self) -> Time:
        """
        Get array of timestamps based on time interval and step size.

        Returns
        -------
        astropy.time.Time
            If begin equals end, returns single timestamp.
            Otherwise returns array of timestamps from begin to end with specified step_size.
        """
        return Time(
            np.arange(self.begin.unix, self.end.unix + self._step_seconds, self._step_seconds), format="unix"
        )

    def _calc(self) -> None:
        """
        Calculate ephemeris data based on the coordinates computed by
        prepare_data().
        """
        if self._rust_ephem is not None:
            # rust-ephem has already computed the angular radii of the Sun,
            # Moon and Earth, as seen from the observatory. The Sun, Moon and
            # Earth are requested as apparent positions (light time,
            # deflection and aberration applied), to match astropy's get_body.
            self.moon = self._rust_ephem.get_body("moon", apparent=True)
            self.sun = self._rust_ephem.get_body("sun", apparent=True)
            self.earth = self.get_body(SolarSystemObject.EARTH)
            self.earth_radius_angle = self._rust_ephem.earth_radius
            self.moon_radius_angle = self._rust_ephem.moon_radius
            self.sun_radius_angle = self._rust_ephem.sun_radius
        else:
            # Calculate the position of the Moon relative to the spacecraft
            self.moon = get_body("moon", self.timestamp, location=self.earth_location)

            # Calculate the position of the Sun relative to the spacecraft
            self.sun = get_body("sun", self.timestamp, location=self.earth_location)

            # Calculate the position of the Earth relative to the spacecraft
            self.earth = get_body("earth", self.timestamp, location=self.earth_location)

            # Calculate Earth's angular radius from observatory, capped at 90 degrees
            self.earth_radius_angle = np.arcsin(np.minimum(R_earth / self.gcrs.distance, 1))

            # Similarly calculate the angular radii of the Sun and the Moon, capped at 90 degrees
            self.moon_radius_angle = np.arcsin(np.minimum(R_moon / self.moon.distance, 1))
            self.sun_radius_angle = np.arcsin(np.minimum(R_sun / self.sun.distance, 1))

        # Get the longitude, latitude, height, and distance (from center of
        # Earth) of the satellite from the EarthLocation object.
        self.longitude = self.earth_location.lon
        self.latitude = self.earth_location.lat
        self.height = self.earth_location.height
        self.distance = self.gcrs.distance

    def get_body(self, body_name: SolarSystemObject | str) -> SkyCoord:
        """
        Return the apparent position of a Solar System body as seen from the
        observatory, for every timestamp in the ephemeris. Results are cached,
        so repeated calls (e.g. when evaluating constraints for many targets)
        are free.

        Parameters
        ----------
        body_name : SolarSystemObject | str
            Name of the body ('mars', 'jupiter', etc.)

        Returns
        -------
        SkyCoord
            Position of the body at each timestamp.
        """
        name = body_name.value if isinstance(body_name, SolarSystemObject) else body_name
        if name not in self._body_cache:
            if self._rust_ephem is not None:
                self._body_cache[name] = self._rust_ephem.get_body(_rust_ephem_body_name(name), apparent=True)
            else:
                self._body_cache[name] = get_body(name, self.timestamp, location=self.earth_location)
        return self._body_cache[name]

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare data for ephemeris calculation. Abstract method, to be implemented by subclasses.
        """
        raise NotImplementedError("prepare_data method must be implemented by subclass")  # pragma: no cover

    def compute(self) -> None:
        """
        Compute ephemeris data.

        This method orchestrates the computation of ephemeris data by first
        preparing the necessary data and then performing the core ephemeris
        calculations. It is intended to be called after initializing the
        Ephemeris object with the desired time range and step size.
        """
        self.prepare_data()
        self._calc()
