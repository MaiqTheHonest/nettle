import math
from time import perf_counter
timer1 = perf_counter()



positions = [[1,5],[3,2],[7,3]]
center_x, center_y = [7,7]
k=1

def TC_dist_to_points(x,y):
    res=0
    for i in positions:
        a, b = i
        res+=math.sqrt((x-a)**2+(y-b)**2)
    res+= k*math.sqrt((x-center_x)**2+(y-center_y)**2)
    return res

inix, iniy = 0,0
cur=TC_dist_to_points(inix, iniy)
step=1
while step > 0.000001:
    f=False
    for dx, dy in [(-1,0), (1,0), (0,1), (0, -1)]:
        curx, cury = inix+step*dx, iniy+ step*dy
        if TC_dist_to_points(curx, cury)<cur:
            cur=TC_dist_to_points(curx, cury)
            inix, iniy = curx, cury
            f=True
        
    if f==False:
        step = step/10
print(cur, [curx, cury])
timer2 = perf_counter()
print(timer2 - timer1)
