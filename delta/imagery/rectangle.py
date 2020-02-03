"""
Simple rectangle class, useful for dealing with ROIs and tiles.
"""
import math

class Rectangle:
    """Simple rectangle class for ROIs. Max values are NON-INCLUSIVE.
       When using it, stay consistent with float or integer values.
    """
    def __init__(self, min_x, min_y, max_x=0, max_y=0,
                 width=0, height=0):
        """Specify width/height by name to use those instead of max_x/max_y."""
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

    def get_bounds(self):
        '''Returns (min_x, max_x, min_y, max_y)'''
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    def width(self):
        return self.max_x - self.min_x
    def height(self):
        return self.max_y - self.min_y

    def has_area(self):
        '''Returns true if the rectangle contains any area.'''
        return (self.width() > 0) and (self.height() > 0)

    def perimeter(self):
        return 2*self.width() + 2*self.height()

    def area(self):
        '''Returns the valid area'''
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

    def make_tile_rois(self, tile_width, tile_height, include_partials=True,
                       overlap_amount=0):
        '''Return a list of tiles encompassing the entire area of this Rectangle'''

        # TODO: Can simplify this a bit!
        tile_spacing_x = tile_width  - overlap_amount
        tile_spacing_y = tile_height - overlap_amount
        num_tiles = (int(math.ceil(self.width()  / tile_spacing_x )),
                     int(math.ceil(self.height() / tile_spacing_y)))
        output_tiles = []
        for c in range(0, num_tiles[0]):
            for r in range(0, num_tiles[1]):

                tile = Rectangle(self.min_x + c*tile_spacing_x,
                                 self.min_y + r*tile_spacing_y,
                                 width=tile_width, height=tile_height)

                if include_partials: # Crop the tile to the valid area and use it
                    tile = tile.get_intersection(self)
                    output_tiles.append(tile)
                else: # Only use it if the uncropped tile fits entirely in this Rectangle
                    if self.contains_rect(tile):
                        output_tiles.append(tile)
        return output_tiles
