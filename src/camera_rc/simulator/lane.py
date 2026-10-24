"""Synthetic lane annotations, independent from road segmentation or OSM lanes.

The corridor is an assumed-width buffer around a reference centreline. The
assumed rectangle fits inside the existing 2.7 m collision disk; this module
does not replace that collision model. Boundary lines are display annotations,
not surveyed lane markings. Perimeter sampling is an explicitly discrete check.
"""
from __future__ import annotations

import math

import numpy as np


class LaneCorridor:
    """A synthetic reference corridor and an explicit assumed vehicle body."""

    SAMPLE_SPACING_M = .1
    END_EXTENSION_M = 20.
    _MAX_DISTANCE_PAIRS = 262144
    _TOLERANCE_M = 1e-9

    def __init__(self, centerline, width_m=3.6, front_m=2.4, rear_m=1.0,
                 vehicle_width_m=1.8, margin_m=.1):
        points = np.asarray(centerline, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError("Centerline must contain at least two finite (x, y) points")
        points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0., axis=1)]].copy()
        if len(points) < 2:
            raise ValueError("Centerline must contain a nonzero length segment")
        for name, value in (("width_m", width_m), ("front_m", front_m), ("rear_m", rear_m),
                            ("vehicle_width_m", vehicle_width_m)):
            if not math.isfinite(value) or value <= 0.:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(margin_m) or margin_m < 0.:
            raise ValueError("margin_m must be finite and nonnegative")
        self.width_m, self.half_width_m = float(width_m), float(width_m)/2.
        self.front_m, self.rear_m = float(front_m), float(rear_m)
        self.vehicle_width_m, self.margin_m = float(vehicle_width_m), float(margin_m)
        direction_start, direction_end = points[1]-points[0], points[-1]-points[-2]
        direction_start /= np.linalg.norm(direction_start)
        direction_end /= np.linalg.norm(direction_end)
        self.centerline = points
        self.extended_centerline = np.vstack((points[0]-self.END_EXTENSION_M*direction_start,
                                             points, points[-1]+self.END_EXTENSION_M*direction_end))
        self.centerline.setflags(write=False)
        self.extended_centerline.setflags(write=False)
        self._starts = self.extended_centerline[:-1]
        self._vectors = np.diff(self.extended_centerline, axis=0)
        self._squared_lengths = np.sum(self._vectors*self._vectors, axis=1)
        self._segment_headings = np.arctan2(self._vectors[:, 1], self._vectors[:, 0])
        half_body = self.vehicle_width_m/2.
        corners = np.array(((self.front_m, half_body), (-self.rear_m, half_body),
                            (-self.rear_m, -half_body), (self.front_m, -half_body)))
        perimeter = []
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            count = int(math.ceil(float(np.linalg.norm(b-a))/self.SAMPLE_SPACING_M))
            perimeter.extend(a + np.arange(count)[:, None]/count*(b-a))
        self.body_offsets = np.asarray(perimeter, dtype=np.float64)
        self.body_offsets.setflags(write=False)
        self._corner_offsets = corners
        self._boundaries = self._make_boundaries()

    def _make_boundaries(self):
        directions = self._vectors/np.sqrt(self._squared_lengths)[:, None]
        normals = np.column_stack((-directions[:, 1], directions[:, 0]))
        offsets = np.empty_like(self.extended_centerline)
        offsets[0], offsets[-1] = normals[0]*self.half_width_m, normals[-1]*self.half_width_m
        for index in range(1, len(offsets)-1):
            bisector = normals[index-1]+normals[index]
            denominator = float(bisector @ normals[index])
            # A reversal has no unique offset join. A bounded display join
            # avoids huge mitres; measurement always uses the true polyline.
            if denominator <= 1e-9:
                offsets[index] = normals[index]*self.half_width_m
            else:
                offset = bisector*self.half_width_m/denominator
                magnitude = float(np.linalg.norm(offset))
                offsets[index] = offset*min(1., 4.*self.half_width_m/magnitude)
        return {"left": (self.extended_centerline+offsets).tolist(),
                "right": (self.extended_centerline-offsets).tolist()}

    @property
    def boundaries(self):
        """Independent display-only synthetic left/right annotation points."""
        return {side: [point[:] for point in points] for side, points in self._boundaries.items()}

    def _nearest(self, points):
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2 or not np.isfinite(values).all():
            raise ValueError("Points must have finite final (x, y) coordinates")
        flat = values.reshape(-1, 2)
        distances = np.empty(len(flat), dtype=np.float64)
        segment_indices = np.empty(len(flat), dtype=np.intp)
        # A segment's axis-aligned box gives a lower distance bound. For each
        # spatially coherent chunk, an exact nearest segment at its box centre
        # plus the box radius gives an upper bound for every contained point.
        # Rejecting boxes above that bound preserves the true minimum while
        # avoiding a full perimeter-point x route-segment projection array.
        ends = self._starts+self._vectors
        boxes_min, boxes_max = np.minimum(self._starts, ends), np.maximum(self._starts, ends)
        coordinate_scale = max(1., float(np.max(np.abs(self.extended_centerline))))
        batch = max(1, min(4096, self._MAX_DISTANCE_PAIRS//len(self._starts)))
        for start in range(0, len(flat), batch):
            chunk = flat[start:start+batch]
            candidates = np.arange(len(self._starts))
            if len(candidates) > 8:
                low, high = np.min(chunk, axis=0), np.max(chunk, axis=0)
                centre = low+(high-low)/2.
                delta = centre-self._starts
                fraction = np.clip(np.sum(delta*self._vectors, axis=1)/self._squared_lengths, 0., 1.)
                residual = delta-fraction[:, None]*self._vectors
                centre_distance = np.sqrt(np.min(np.sum(residual*residual, axis=1)))
                # Padding only keeps extra candidates; it does not change the
                # measured distance or the public lane boundary tolerance.
                scale = max(coordinate_scale, float(np.max(np.abs(chunk))))
                padding = self._TOLERANCE_M+32.*np.finfo(np.float64).eps*scale
                bound = centre_distance+np.linalg.norm((high-low)/2.)+padding
                gap = np.maximum(np.maximum(boxes_min-high, low-boxes_max), 0.)
                candidates = np.flatnonzero(np.sum(gap*gap, axis=1) <= bound*bound)
            starts, vectors = self._starts[candidates], self._vectors[candidates]
            dx = chunk[:, 0, None]-starts[None, :, 0]
            dy = chunk[:, 1, None]-starts[None, :, 1]
            along = (dx*vectors[None, :, 0]+dy*vectors[None, :, 1])/self._squared_lengths[candidates]
            np.clip(along, 0., 1., out=along)
            dx -= along*vectors[None, :, 0]
            dy -= along*vectors[None, :, 1]
            squared = dx*dx + dy*dy
            nearest = np.argmin(squared, axis=1)
            distances[start:start+len(chunk)] = np.sqrt(squared[np.arange(len(chunk)), nearest])
            # flatnonzero retains original segment order, including exact ties
            # used by stationary path poses to choose a reference heading.
            segment_indices[start:start+len(chunk)] = candidates[nearest]
        return distances.reshape(values.shape[:-1]), segment_indices.reshape(values.shape[:-1])

    def distance_to_centerline(self, points):
        """Nearest distance to the endpoint-extended synthetic centreline."""
        return self._nearest(points)[0]

    @staticmethod
    def _transform(offsets, positions, headings):
        cosine, sine = np.cos(headings)[:, None], np.sin(headings)[:, None]
        result = np.empty((len(positions), len(offsets), 2), dtype=np.float64)
        result[:, :, 0] = positions[:, 0, None]+cosine*offsets[None, :, 0]-sine*offsets[None, :, 1]
        result[:, :, 1] = positions[:, 1, None]+sine*offsets[None, :, 0]+cosine*offsets[None, :, 1]
        return result

    def sample_points(self, x, y, yaw):
        """World perimeter points, including corners, spaced at most 0.1 m."""
        values = np.asarray((x, y, yaw), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Vehicle pose must be finite")
        return self._transform(self.body_offsets, values[None, :2], values[None, 2])[0]

    def measure(self, x, y, yaw):
        """Measure actual body departure; clearance excludes the guard margin."""
        perimeter = self.sample_points(x, y, yaw)
        offsets = self.distance_to_centerline(perimeter)
        maximum = float(np.max(offsets))
        center = float(self.distance_to_centerline(np.array((x, y))))
        clearance = self.half_width_m-maximum
        return {"outside": bool(maximum > self.half_width_m+self._TOLERANCE_M),
                "center_offset_m": center, "max_body_offset_m": maximum,
                "clearance_m": clearance}

    def allows_path(self, world_points, headings=None):
        """Require each sampled rectangle plus margin inside the corridor.

        Without headings, forward path differences determine vehicle headings;
        the final pose reuses the preceding direction. A stationary/isolated
        pose falls back to its closest reference-segment heading.
        """
        points = np.asarray(world_points, dtype=np.float64)
        if points.size == 0:
            return False
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Path must contain (x, y) points")
        if not np.isfinite(points).all():
            return False
        if headings is None:
            _, nearest = self._nearest(points)
            angles = self._segment_headings[nearest].copy()
            if len(points) > 1:
                delta = np.diff(points, axis=0)
                moving = np.any(delta != 0., axis=1)
                angles[:-1][moving] = np.arctan2(delta[moving, 1], delta[moving, 0])
                if moving[-1]:
                    angles[-1] = angles[-2]
        else:
            angles = np.asarray(headings, dtype=np.float64)
            if angles.ndim != 1 or len(angles) != len(points):
                raise ValueError("One heading is required for every path point")
            if not np.isfinite(angles).all():
                return False
        limit = self.half_width_m-self.margin_m+self._TOLERANCE_M
        if limit < 0.:
            return False
        # Corner rejection is fast for clearly invalid candidates. The full
        # perimeter check still runs when corners fit, including at bends where
        # the centreline buffer is not convex. Batch allocation stays bounded.
        for start in range(0, len(points), 128):
            poses, directions = points[start:start+128], angles[start:start+128]
            corners = self._transform(self._corner_offsets, poses, directions)
            if np.any(self.distance_to_centerline(corners) > limit):
                return False
            perimeter = self._transform(self.body_offsets, poses, directions)
            if np.any(self.distance_to_centerline(perimeter) > limit):
                return False
        return True


def road_lane_layout(lane,opposing_width_m=3.6):
    """Assumed right-hand two-lane layout; the existing route is the ego lane."""
    inner=lane.boundaries
    outer=LaneCorridor(lane.centerline,width_m=2*(lane.half_width_m+opposing_width_m)).boundaries['left']
    opposite=LaneCorridor(lane.centerline,width_m=2*(lane.half_width_m+opposing_width_m/2)).boundaries['left']
    return {'source':'synthetic_lane_assignment_not_measured','ego_lane_id':'ego_forward',
        'lane_count':2,'divider':inner['left'],'ego_outer_boundary':inner['right'],
        'opposing_outer_boundary':outer,'lanes':[
            {'id':'ego_forward','direction':'forward','width_m':lane.width_m,'is_ego':True,'centerline':lane.extended_centerline.tolist()},
            {'id':'opposing_reverse','direction':'reverse','width_m':opposing_width_m,'is_ego':False,'centerline':opposite[::-1]}]}
