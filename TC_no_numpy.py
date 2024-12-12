from math import sqrt
from time import perf_counter
timer1 = perf_counter()



#points = [[1,5],[3,2],[7,3]]

def weighted_distance(points, center, weight=1) -> list:

    center_x, center_y = center

    def TC_dist_to_points(x,y):
        res=0
        for point in points:
            a, b = point[0]
            res+=sqrt((x-a)**2+(y-b)**2)                # sums the point-centroid differences
        res+= weight*sqrt((x-center_x)**2+(y-center_y)**2)   # sums the centroid-center difference
        return res

    step=1
    inix, iniy = 0,0

    TC_now = TC_dist_to_points(inix, iniy)

    while step > 0.000001:
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

#print(weighted_distance(points=points, center=[7,7], k=1))

# print(TC_now ,  [curx, cury])
# timer2 = perf_counter()
# print(timer2 - timer1)
