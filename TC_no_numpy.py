from math import sqrt



def weighted_distance(points, center=None, weight=1) -> list:

    def TC_dist_to_points(x,y):
        res=0
        for point in points:
  
            a, b = point
            res+=sqrt((x-a)**2+(y-b)**2)                     # sums the point-centroid differences


        if center is not None:
            center_x, center_y = center
            res+= weight*sqrt((x-center_x)**2+(y-center_y)**2)   # sums the centroid-center difference
        return res

    step=1
    inix, iniy = 0,0

    TC_now = TC_dist_to_points(inix, iniy)

    while step > 0.00001:
        f=False
        for dx, dy in [(-1,0), (1,0), (0,1), (0, -1)]:
            curx, cury = inix+step*dx, iniy+ step*dy
            if TC_dist_to_points(curx, cury)<TC_now : 
                TC_now = TC_dist_to_points(curx, cury)
                inix, iniy = curx, cury
                f=True
            
        if f==False:
            step = step/10

    return [curx, cury, TC_now]

