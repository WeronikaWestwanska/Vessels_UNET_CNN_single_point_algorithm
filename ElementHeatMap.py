import numpy

class RadialElement(object):
    """Class defines circle of a specific diameter around
    the element's (object or background) central points and returns an in a given array"""

    #---------------------------------------------------------------
    # ctor:
    # min_prob - minimum element probability
    # max_prob - maximum element probability
    # radius - radius of the vessel circle
    #---------------------------------------------------------------
    def __init__(self, min_prob = 0.6, max_prob = 1.0, radius = 40):

        self.min_prob = min_prob
        self.max_prob = max_prob
        self.radius = radius
        self.radius_dict = dict()

        # create dictionary with x,y -> radius calculations to speed up
        # data generation
        radiusf = float(self.radius)
        for x in range(0, self.radius + 1):          
          xf = float(x)
          xf *= xf          
          for y in range(0, self.radius + 1):            
            yf = float(y)
            yf *= yf            
            vessel_probability = 0.0
            current_radius = numpy.sqrt(xf + yf)
            if current_radius <= radiusf:
                # check if lies in circle
                vessel_probability = self.max_prob - (self.max_prob - self.min_prob) * (current_radius / radiusf)

            self.radius_dict[(y, x)] = vessel_probability

class ElementHeatMap(object):
    """Class which draws circles of a specific diameter around
    the elements central point and returns it in a given array"""

    #---------------------------------------------------------------
    # ctor:
    # radial_element - model of an element
    # elements_positions_list - list with positions of elements
    # height - height of the picture
    # width - width of the picture
    #---------------------------------------------------------------
    def __init__(self,
                 radial_element,
                 elements_positions_list,
                 height,
                 width):

        self.radial_element = radial_element
        self.elements_positions_list = elements_positions_list
        self.height = height
        self.width = width
            
    #---------------------------------------------------------------
    # get the heatmap for the vessels as a 2 dimensional array,
    # with values from 0.0 to 1.0
    #---------------------------------------------------------------
    def get_heatmap(self):
        
        # declare result heatmap
        self.heatmap_array = numpy.zeros((self.height, self.width))
        
        # iterate over the elements
        for element_position in self.elements_positions_list:

            # analyse each element separately
            self.get_heatmap_per_element(element_position)

        return self.heatmap_array
               
    #---------------------------------------------------------------
    # get the heatmap for a vessel, by filling existing 
    # 2 dimensional array, with values from 0.0 to 1.0
    # if elements overlap, then max cannot be greater than 1.0
    # heatmap_array - existing elements
    # element_position - specific element position
    #---------------------------------------------------------------
    def get_heatmap_per_element(self, element_position):
      
        x_start, y_start = element_position

        # draw circle 
        xmin = x_start - self.radial_element.radius
        xmin = numpy.maximum(0, xmin)
        xmax = x_start + self.radial_element.radius
        xmax = numpy.minimum(self.width - 1, xmax)

        ymin = y_start - self.radial_element.radius
        ymin = numpy.maximum(0, ymin)
        ymax = y_start + self.radial_element.radius
        ymax = numpy.minimum(self.height - 1, ymax)

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):

                xo = numpy.abs(x_start - x)
                yo = numpy.abs(y_start - y)
                element_probability = self.radial_element.radius_dict[(yo, xo)]

                self.heatmap_array[y][x] += element_probability
                if self.heatmap_array[y][x] > 1.0:
                    self.heatmap_array[y][x] = 1.0

