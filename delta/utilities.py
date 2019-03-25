#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __BEGIN_LICENSE__
#  Copyright (c) 2009-2013, United States Government as represented by the
#  Administrator of the National Aeronautics and Space Administration. All
#  rights reserved.
#
#  The NGT platform is licensed under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance with the
#  License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# __END_LICENSE__

"""
Miscellaneous utility classes/functions.
"""

import gdal



#============================================================================
# Constants

BYTES_PER_MB = 1024*1024

#============================================================================
# Functions

def get_num_bytes_from_gdal_type(gdal_type):
    """Return the number of bytes for one pixel (one band) in a GDAL type."""
    results = {
        gdal.GDT_Byte:    1,
        gdal.GDT_UInt16:  2,
        gdal.GDT_UInt32:  4,
        gdal.GDT_Float32: 4
    }
    return results.get(gdal_type)

#============================================================================
# Classes

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
        #
        #if not self.hasArea(): # Debug helper
        #    print 'RECTANGLE WARNING: ' + str(self)
    
    def __str__(self):
        return ('min_x: %f, max_x: %f, min_y: %f, max_y: %f' %
                (self.min_x, self.max_x, self.min_y, self.max_y))
    
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
        else:
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
        if yScale == None:
            yScale = xScale
        self.min_x *= xScale
        self.max_x *= xScale
        self.min_y *= yScale
        self.max_y *= yScale
        
    def expand(self, left, down, right=None, up=None):
        '''Expand the box by an amount in each direction'''
        self.min_x -= left
        self.min_y -= down
        if right == None: # If right and up are not passed in, use left and down for both sides.
            right = left
        if up == None:
            up = down
        self.max_x += right
        self.max_y += up

    def expand_to_contain(self, x, y):
        '''Expands the rectangle to contain the given point'''
        if isinstance(self.min_x, float):
            delta = 0.001
        else: # These are needed because the upper bound is non-exclusive.
            delta = 1
        if x < self.min_x: self.min_x = x
        if y < self.min_y: self.min_y = y
        if x > self.max_x: self.max_x = x + delta
        if y > self.max_y: self.max_y = y + delta
        
    def get_intersection(self, other_rect):
        '''Returns the overlapping region of two rectangles'''
        overlap = Rectangle(max(self.min_x, other_rect.min_x),
                            max(self.min_y, other_rect.min_y),
                            min(self.max_x, other_rect.max_x),
                            min(self.max_y, other_rect.max_y))
        return overlap
        
    def contains(self, other_rect):
        '''Returns true if this rect contains all of the other rect'''
        if self.min_x > other_rect.min_x: return False
        if self.min_y > other_rect.min_y: return False
        if self.max_x < other_rect.max_x: return False
        if self.max_y < other_rect.max_y: return False
        return True
        
    def overlaps(self, other_rect):
        '''Returns true if there is any overlap between this and another rectangle'''
        overlap_area = self.get_intersection(other_rect)
        return overlap_area.hasArea()
    


    
    
    
    
    
    
    
    
