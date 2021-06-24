# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple rectangle class, useful for dealing with ROIs and tiles.
"""
import math
import copy

from numpy.lib.shape_base import tile

class Rectangle:
    """
    Simple rectangle class for ROIs. Max values are NON-INCLUSIVE.
    When using it, stay consistent with float or integer values.
    """
    def __init__(self, min_x, min_y, max_x=0, max_y=0,
                 width=0, height=0):
        """
        Parameters
        ----------
        min_x: int
        min_y: int
        max_x: int
        max_y: int
            Rectangle bounds.
        width: int
        height: int
            Specify width / height to use these instead of max_x/max_y.
        """
        self.min_x = min_x
        self.min_y = min_y
        if width > 0:
            self.max_x = min_x + width
        else:
            self.max_x = max_x
        if height > 0:
            self.max_y = min_y + height
        else:
            self.max_y = max_y

    def __str__(self):
        if isinstance(self.min_x, int):
            return ('min_x: %d, max_x: %d, min_y: %d, max_y: %d' %
                    (self.min_x, self.max_x, self.min_y, self.max_y))
        return ('min_x: %f, max_x: %f, min_y: %f, max_y: %f' %
                (self.min_x, self.max_x, self.min_y, self.max_y))

    def __repr__(self):
        return self.__str__()

#    def indexGenerator(self):
#        '''Generator function used to iterate over all integer indices.
#           Only use this with integer boundaries!'''
#        for row in range(self.min_y, self.max_y):
#            for col in range(self.min_x, self.max_x):
#                yield(TileIndex(row,col))

    def bounds(self):
        """
        Returns
        -------
        (int, int, int, int):
            (min_x, max_x, min_y, max_y)
        """
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    def width(self):
        return self.max_x - self.min_x
    def height(self):
        return self.max_y - self.min_y

    def has_area(self):
        """
        Returns
        -------
        bool:
            true if the rectangle contains any area.
        """
        return (self.width() > 0) and (self.height() > 0)

    def perimeter(self):
        return 2*self.width() + 2*self.height()

    def area(self):
        if not self.has_area():
            return 0
        return self.height() * self.width()

    def get_min_coord(self):
        return (self.min_x, self.min_y)
    def get_max_coord(self):
        return (self.max_x, self.max_y)

    def shift(self, dx, dy):
        '''Shifts the entire box'''
        self.min_x += dx
        self.max_x += dx
        self.min_y += dy
        self.max_y += dy

    def scale_by_constant(self, xScale, yScale):
        '''Scale the units by a constant'''
        if yScale is None:
            yScale = xScale
        self.min_x *= xScale
        self.max_x *= xScale
        self.min_y *= yScale
        self.max_y *= yScale

    def expand(self, left, down, right=None, up=None):
        '''Expand the box by an amount in each direction'''
        self.min_x -= left
        self.min_y -= down
        if right is None: # If right and up are not passed in, use left and down for both sides.
            right = left
        if up is None:
            up = down
        self.max_x += right
        self.max_y += up

    def expand_to_contain_pt(self, x, y):
        '''Expands the rectangle to contain the given point'''
        if isinstance(self.min_x, float):
            delta = 0.001
        else: # These are needed because the upper bound is non-exclusive.
            delta = 1
        if x < self.min_x: self.min_x = x
        if y < self.min_y: self.min_y = y
        if x > self.max_x: self.max_x = x + delta
        if y > self.max_y: self.max_y = y + delta

    def expand_to_contain_rect(self, other_rect):
        '''Expands the rectangle to contain the given rectangle'''

        if other_rect.min_x < self.min_x: self.min_x = other_rect.min_x
        if other_rect.min_y < self.min_y: self.min_y = other_rect.min_y
        if other_rect.max_x > self.max_x: self.max_x = other_rect.max_x
        if other_rect.max_y > self.max_y: self.max_y = other_rect.max_y

    def get_intersection(self, other_rect):
        '''Returns the overlapping region of two rectangles'''
        overlap = Rectangle(max(self.min_x, other_rect.min_x),
                            max(self.min_y, other_rect.min_y),
                            min(self.max_x, other_rect.max_x),
                            min(self.max_y, other_rect.max_y))
        return overlap

    def contains_pt(self, x, y):
        '''Returns true if this rect contains the given point'''
        if self.min_x > x: return False
        if self.min_y > y: return False
        if self.max_x < x: return False
        if self.max_y < y: return False
        return True

    def contains_rect(self, other_rect):
        '''Returns true if this rect contains all of the other rect'''
        if self.min_x > other_rect.min_x: return False
        if self.min_y > other_rect.min_y: return False
        if self.max_x < other_rect.max_x: return False
        if self.max_y < other_rect.max_y: return False
        return True

    def overlaps(self, other_rect):
        '''Returns true if there is any overlap between this and another rectangle'''
        overlap_area = self.get_intersection(other_rect)
        return overlap_area.has_area()

    def make_tile_rois(self, tile_shape, overlap_shape=(0, 0), include_partials=True, min_shape=(0, 0),
                       partials_overlap=False, by_block=False, containing_rect=None):
        """
        Return a list of tiles encompassing the entire area of this Rectangle.

        Parameters
        ----------
        tile_shape: (int, int)
            Shape of each tile (width, height)
        overlap_shape: (int, int)
            Amount to overlap tiles in x and y direction
        include_partials: bool
            If true, include partial tiles at the edge of the image.
        min_shape: (int, int)
            If true and `partials` is true, keep partial tiles of this minimum size.
        partials_overlap: bool
            If `partials` is false, and this is true, expand partial tiles
            to the desired size. Tiles may overlap in some areas.
        by_block: bool
            If true, changes the returned generator to group tiles by block.
            This is intended to optimize disk reads by reading the entire block at once.
        containing_rect: Rectangle
            Tiles are restricted to fit inside this rectangle instead of "self".

        Returns
        -------
        List[Rectangle]:
            Generator yielding ROIs. If `by_block` is true, returns a generator of (Rectangle, List[Rectangle])
            instead, where the first rectangle is a larger block containing multiple tiles in a list.
        List[Rectangle]:
            Same as the first output, but the ROIs do not include any overlap regions or any area outside the rectangle.
        """
        tile_width, tile_height = tile_shape
        min_width, min_height = min_shape
        if not containing_rect:
            containing_rect = self

        half_overlap = (overlap_shape[0] // 2, overlap_shape[1] // 2)
        tile_spacing_x = tile_width  - overlap_shape[0]
        tile_spacing_y = tile_height - overlap_shape[1]
        num_tiles_round_up = (int(math.ceil((self.width() - overlap_shape[0]) / tile_spacing_x)),
                              int(math.ceil((self.height()- overlap_shape[1]) / tile_spacing_y)))
        if include_partials or partials_overlap:
            num_tiles = num_tiles_round_up
        else:
            containing_rect_tiles = (int(math.floor((containing_rect.width() - overlap_shape[0]) / tile_spacing_x)),
                                     int(math.floor((containing_rect.height()- overlap_shape[1]) / tile_spacing_y)))
            num_tiles = (min(num_tiles_round_up[0], containing_rect_tiles[0]),
                         min(num_tiles_round_up[1], containing_rect_tiles[1]))

        output_tiles = []
        unique_tiles = []
        for c in range(0, num_tiles[0]):
            row_tiles = []
            unique_row_tiles = []
            for r in range(0, num_tiles[1]):
                tile = Rectangle(self.min_x + c*tile_spacing_x,
                                 self.min_y + r*tile_spacing_y,
                                 width=tile_width, height=tile_height)

                # The unique tile has overlap regions removed
                # and is always constrained by the rectangle size
                unique_tile = copy.copy(tile)
                if c > 0:
                    unique_tile.min_x += half_overlap[0]
                if r > 0:
                    unique_tile.min_y += half_overlap[1]
                if c < num_tiles[0]-1:
                    unique_tile.max_x -= half_overlap[0]
                if r < num_tiles[1]-1:
                    unique_tile.max_y -= half_overlap[1]
                unique_tile = unique_tile.get_intersection(self)

                if include_partials: # Crop the tile to the valid area and use it
                    tile = tile.get_intersection(self)
                    if tile.width() < min_width or tile.height() < min_height:
                        continue
                else: # Only use it if the uncropped tile fits entirely in this Rectangle
                    if not containing_rect.contains_rect(tile):
                        if not partials_overlap:
                            continue
                        # Try shifting tile "back" in x/y so that we can fit the entire proposed tile
                        new_max_x = tile.max_x
                        new_max_y = tile.max_y
                        if tile.max_x > containing_rect.max_x:
                            new_max_x = min(containing_rect.max_x, tile.max_x)
                        if tile.max_y > containing_rect.max_y:
                            new_max_y = min(containing_rect.max_y, tile.max_y)

                        tile = Rectangle(new_max_x - tile_width, new_max_y - tile_height,
                                         width=tile_width, height=tile_height)
                        if not containing_rect.contains_rect(tile):
                            continue
                if by_block:
                    row_tiles.append(tile)
                    unique_row_tiles.append(unique_tile)
                else:
                    output_tiles.append(tile)
                    unique_tiles.append(unique_tile)

            if by_block and row_tiles:
                row_rect = Rectangle(row_tiles[0].min_x, row_tiles[0].min_y,
                                     row_tiles[-1].max_x, row_tiles[-1].max_y)
                for r in row_tiles:
                    r.shift(-row_rect.min_x, -row_rect.min_y)
                for r in unique_row_tiles:
                    r.shift(-row_rect.min_x, -row_rect.min_y)
                output_tiles.append((row_rect, row_tiles))
                unique_tiles.append((row_rect, unique_row_tiles))

        return output_tiles, unique_tiles

    def make_tile_rois_yx(self, tile_shape, overlap_shape=(0, 0), include_partials=True, min_shape=(0, 0),
                          partials_overlap=False, by_block=False, containing_rect=None):
        '''As make_tile_rois but using a (y,x) input format instead of (x,y)'''
        return self.make_tile_rois((tile_shape[1], tile_shape[0]),
                                   (overlap_shape[1], overlap_shape[0]),
                                   include_partials, (min_shape[1], min_shape[0]),
                                   partials_overlap, by_block, containing_rect)
