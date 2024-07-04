"""
Adapted from D. Rein-Weston

2D example of aircraft at (x0,y0) with destination of (xdest,ydest),
and assumed constant speed and altitude (parameters defined in "odfile.txt").
Straight-line route is created with waypoints spaced by an average number
of nautical miles (defined in the parse function in "mytools.py").
Obstacles of varying number of vertices are read-in from obstacle definition
(eg, SC1_multi_convex.txt) file. Line segments that define obstacles are
checked to see if there are any intersections with route segments.
The branching algorithm proceeds by branching on first encountered obstacle,
creating clockwise and counterclockwise route alternatives, and selecting
the most "promising" route (i.e. the route requiring least deviation from its
parent route) first for further branching. Less-promising routes are later
returned to and evaluated. The most efficient route option is plotted.

Input settings include windsOn parameter and optimizationpriority parameter.
By setting the optimization priority to 0 or 1, the algorithm evaluates a
distance-efficient or time-efficient route, respectively.

Reads from origin/destination and obstacle definition text files
Uses tools from "mytools.py" and "windfield.py".

"""

import matplotlib.pyplot as plt
import numpy as np
from bluesky_gym.envs.common.tools_deterministic_path_planning import Pos, LatLon2XY, XY2LatLon, processread,Obs,parse,obsReader,\
Route,wyptannotate,callWinds,intersectionpt,specifywindfield

#from math import degrees
#import math.isinf as infinity
#import random as rand
#import copy

# from mytoolsFINAL 
# from openfile import filedialog

#INPUT SETTINGS TO DEFINE

#################################################################
# Define text files to read-in
#origin_dest_speed_file  = "SC3_odfile.txt"
#obstacle_file           = "SC3_pacman.txt"

# obstacle_file            = filedialog("Select Obstacle file","./Obstacles")
# origin_dest_speed_file   = filedialog("Select Route file","./Routes")
def det_path_planning(lat0, lon0, altitude, TAS, latdest, londest, inputObs):

    # define whether or not to use wind
    windsOn                 = 0
    # Define optimization strategy
    optimizationpriority    = 0       # type 0 for distance, 1 for time
    #################################################################

    #PLOTTING OPTIONS

    #################################################################
    # plot direct route
    pltdirectrt             = 0
    # plot all trial solutions
    plttrialsols            = 0
    # plot final incumbent
    pltfinalsol             = 0
    #################################################################

    # READ-IN & INTERPRET AVAILABLE PARAMETERS

    #################################################################
    # DEFINE ORIGIN, DESTINATION, SPEED

    # read origin and destination lat/lon from pre-defined text file
    # lst =                   processread(origin_dest_speed_file)
    #process the data:
    # (lat0,lon0) =           lst[0]                          # initial latitude and longitude
    # (latdest,londest) =     lst[1]                          # destination latitude and longitude

    # # process third input as altitude, note: altitude taken as constant throughout trajectory
    # altitude =              lst[2][0]/3.28084               # altitude converted to [m] for windfield.py
    # # process fourth input as true airspeed (TAS), note: TAS taken as constant throughout trajectory
    # TAS =                   lst[2][1]

    # convert origin and destination to x,y coordinates
    origin =                Pos(LatLon2XY(lat0,lon0))
    destination =           Pos(LatLon2XY(latdest,londest))

    # DEFINE OBSTACLE(S)

    # read obstacle defintions from pre-defined text file
    # inputObs =              obsReader(obstacle_file)  

    # While processing obstacle vertices in for loops, also define
    # boundaries of windfield
    orig,dest = [lat0, lon0],[latdest,londest]
    ymin = min(orig[0],dest[0])
    ymax = max(orig[0],dest[0])
    xmin = min(orig[1],dest[1])
    xmax = max(orig[1],dest[1])
    
    obstacle_list_xy = []
    # for each obstacle in the dictionary:
    for i in range(len(inputObs)):
        vertices_list_xy = []
        for j in range(len(inputObs[i])):
            (obY,obX) = inputObs[i][j]
            ymin = min(ymin,obY)
            ymax = max(ymax,obY)
            xmin = min(xmin,obX)
            xmax = max(xmax,obX)
            # convert vertices to XY coordinates
            vertices_list_xy.append(LatLon2XY(obY,obX))
        obstacle_list_xy.append(vertices_list_xy)

    # create obstacle dictionary (key is obstacle index)
    obsDic_xy = {i: obstacle_list_xy[i] for i in range(len(obstacle_list_xy))}
    # print(obsDic_xy)

    # define as instance of the obstacle class
    for i in range(len(obsDic_xy)):
        obsDic_xy[i] = Obs(obsDic_xy[i])

    # Add edge around it of 1%
    margin = 0.01
    xspan = xmax-xmin
    yspan = ymax-ymin

    xmin = xmin - margin*xspan
    xmax = xmax + margin*xspan
    ymin = ymin - margin*yspan
    ymax = ymax + margin*yspan
    
    wind = specifywindfield(ymin,ymax,xmin,xmax)

    #################################################################

    # GENERATE DIRECT ROUTE and PRINT ROUTE INFO

    #################################################################

    # generate direct route using origin and destination info
    distance, heading, wypts = parse(origin,destination)
    nominaltime = distance/TAS
    directrt = Route(origin,destination,TAS,wypts,distance,nominaltime,0.0)

    # print distance and heading info
    print('direct route info:')
    print('distance       ', distance, ' [NM]')
    print('true airspeed  ', TAS, '         [kts]')
    print('time (no wind) ', nominaltime, '[hr]')
    #print 'heading is', degrees(heading), 'degrees'
    print('')
    if optimizationpriority == 0:
        print('optimizing for DISTANCE')
    elif optimizationpriority == 1:
        print('optimizing for TIME')
    print('')
    print('############## branching algorithm ##############')
    print('')

    # REMOVE DIRECT ROUTE WAYPOINTS THAT ARE INSIDE ANY OBSTACLE

    directrt.clean(obsDic_xy)

    Route.active[-1] = directrt

    #################################################################


    # MAKE PLOT SHOWING DIRECT ROUTE AND OBSTACLES

    #################################################################

    # create plot
    fig = plt.figure()

    # define waypoints along direct route, annotate and plot
    (xwpts,ywpts) = list(zip(*directrt.waypoints))               # unzip x and y waypoint coordinates
    wyptannotate(xwpts,ywpts)
    #plt.plot(xwpts,ywpts,'--')

    # label figure
    fig.suptitle('2D Avoidance', fontsize=20)
    ax = fig.axes[0]
    plt.xlabel('X: position w.r.t. Prime Meridian [NM]')
    # plt.xlim([])
    plt.ylabel('Y: position w.r.t. Equator [NM]')

    # for all obstacles: draw and label (with a number according to order defined)
    for cur in range(len(obsDic_xy)):
        obsDic_xy[cur].plotter(fig,cur)

    #################################################################


    # IF INTERSECTION OF ROUTE SEGMENTS WITH OBSTACLE SEGMENTS EXISTS,
    # CREATE ALTERNATIVE ROUTES

    #################################################################

    incflag = 0
    incumbent = float('inf')

    while len(Route.active): 
        print('length of active: ',len(Route.active))
        if len(Route.active) > 500:
            import code
            code.interact(local= locals())
        directrtplted = 0
        
        ## logic for setting parent to least deviation OR only option from active list
        if len(Route.active) == 1:
            parent = Route.active.pop()
            # if there is only 1 active route and no incumbent,
            # assume it's direct rt and plot for reference        
            if incflag == 0 and pltdirectrt:
                parentX,parentY = list(zip(*parent.waypoints))
                plt.plot(parentX,parentY,'--') 
                directrtplted = 1
        else:
            deviationmin = []
            for i in range(len(Route.active)):
                deviationmin.append(Route.active[i].deviation)
            topop = deviationmin.index(min(deviationmin))
            parent = Route.active.pop(topop)
        ##
        if plttrialsols and not directrtplted:    
            parentX,parentY = list(zip(*parent.waypoints))
            plt.plot(parentX,parentY,'--')

        # print('parentX', parentX)
        # print('parentY', parentY)
        # if there is an incumbent... check if route length (or time) is greater than incumbent length (or time)
        if incflag==1:
            if optimizationpriority == 0:
                if parent.distance>=incumbent.distance:
                    # "fathom route" because it can only get longer with deviations
                    print('> incumbent')
    #                print 'parent distance [NM]:', parent.distance
                    print('')
                    continue
            elif optimizationpriority == 1:
                if parent.time>=incumbent.time:
                    # "fathom route" because it can only get longer with deviations
                    print('> incumbent')
    #                print 'parent time [hr]:', parent.time
                    print('')
    #                parentX,parentY = zip(*parent.waypoints)
    #                plt.plot(parentX,parentY,'-.')   
                    continue

    #    Parentx,Parenty = zip(*parent.waypoints)
    #    plt.plot(Parentx,Parenty,'--')     


        # check if parent route intersects any obstacles
        allintersections =  []                          # create empty list to hold lists for each obstacle       
        for obstacle in range(len(obsDic_xy)):             # loop through all obstacles
            # tabulate intersections between route segments and obstacle segments
            allintersections.append(obsDic_xy[obstacle].intersectroute(parent))
        # black(allintersections)
        # if there are intersections, define branching obstacle as first encountered
        if len([_f for _f in allintersections if _f]):
            # find index of obstacle in obsDic that will be encountered first
            firstseg = []
            for obst in range(len(allintersections)):
                rseg = []
                if allintersections[obst]:  
                    for i in range(len(allintersections[obst])):
                        rseg.append(allintersections[obst][i][0])
                    firstseg.append(min(rseg))
                else:
                    firstseg.append('NaN')
            
            # find index of minimum route segment in firstseg list
            
            # if the minimum value is in the list multiple times, calculate
            # intersection pts of each obstacle within that route segment and
            # select obstacle with least distance between route segment start and
            # point of intersection with obstacle
            

            first_seg = [f for f in firstseg if isinstance(f, int)]
            # red(firstseg)
            if firstseg.count(min(first_seg)) > 1:

                indices = [i for i, x in enumerate(firstseg) if x == min(first_seg)]
                distlist = []
                options = []
                refpt = Pos(parent.waypoints[min(first_seg)])
                routpt1 = parent.waypoints[min(first_seg)]
                routpt2 = parent.waypoints[min(first_seg)+1]
                
                for i in range(len(indices)):
                    for j in range(len(allintersections[indices[i]])):
                        # green(indices[i])
                        # red(allintersections[indices[i]])
                        # yellow(allintersections[indices[i]][j])
                        # cyan(allintersections[indices[i]][j][1]+1)
                        # print(len(obsDic_xy[indices[i]].vert))
                        a = obsDic_xy[indices[i]].vert[allintersections[indices[i]][j][1]]
                        if len(obsDic_xy[indices[i]].vert) == (allintersections[indices[i]][j][1]+1):
                            b = obsDic_xy[indices[i]].vert[0]
                        else:
                            b = obsDic_xy[indices[i]].vert[allintersections[indices[i]][j][1]+1]
                        checkdist = intersectionpt(a,b,routpt1,routpt2) - refpt                  
                        distlist.append((checkdist.length()))               
                    options.append(min(distlist))
                
                branch = indices[options.index(min(options))]

            # else just take index of minimum route segment in firstseglist
            else:
                branch = firstseg.index(min(first_seg))
                
            # print which obstacle has been selected as branching obstacle
            print('branch obstacle', branch)
            print('')
            
            intersectTab = allintersections[branch]
        
            # (A) populate list of obstacle segments that intersect with route segment
            segList = []                                           # empty list
            for index in range(len(intersectTab)):
                segList.append(intersectTab[index][1])             # segList contains the smaller of the two obstacle indices defining the segment
            #     yellow(segList[-1])
            # green(segList)
            # (B) re-sort obstacle segment list here, in order of first-encountered       
            segList = obsDic_xy[branch].resort(segList,parent,intersectTab)

            # (C) populate lists of left and right alternative waypoints
            # (make use of the fact that obstacle vertices are defined in clockwise order)
        
            altWptL = obsDic_xy[branch].leftalt(segList)
            altWptR = obsDic_xy[branch].rightalt(segList)

    #        altWptLclean = altWptL
    #        altWptRclean = altWptR
            altWptLclean = obsDic_xy[branch].extupdate(altWptL,destination)
            altWptRclean = obsDic_xy[branch].extupdate(altWptR,destination)
            

            # (D) CREATE TRIAL ROUTES WITH PREVIOUSLY CALCULATED ALTERNATIVE WAYPOINTS
            # check which route segment contained the first intersection, insert alternative waypoints       
            altWptIndex = intersectTab[0][0] + 1                    # note: this works because no obstacle will span more than one route segment

            # ROUTE L
            routeL = Route(origin,destination,TAS,parent.waypoints,parent.distance,parent.time,parent.deviation)
            routeL.insert(altWptIndex,altWptLclean)

            # backward cleanup
            lstpt = altWptIndex+len(altWptLclean)                  # define the last waypoint index to consider (exit wypt from branching obstacle)
            routeL.backwardcleanup(obsDic_xy,lstpt)

            if windsOn == 1:
                windsaloftL = callWinds(wind,routeL) 
            else:
                windsaloftL = [np.zeros(len(routeL.waypoints)),np.zeros(len(routeL.waypoints))]

            routeL.deviation = routeL.deviationcheck(parent,optimizationpriority,windsaloftL)

            # should consider somehow adding the alt waypoints (and doing any other necessary changes, i.e. backward cleanup)
            # and THEN declaring it's a route (and thus adding it to active waypoint list)
            # need to do this as well for the routeR and directrt
            Route.active[-1] = routeL
            
            # plot left route in red        
    #        leftpltX,leftpltY = zip(*routeL.waypoints)
    #        plt.plot(leftpltX,leftpltY,':',color='r')


            # ROUTE R        
            routeR = Route(origin,destination,TAS,parent.waypoints,parent.distance,parent.time,parent.deviation)
            routeR.insert(altWptIndex,altWptRclean)

            # backward cleanup
            lstpt = altWptIndex+len(altWptRclean)                  # define the last waypoint index to consider (exit wypt from branching obstacle)        
            routeR.backwardcleanup(obsDic_xy,lstpt)
    
            
            if windsOn == 1:
                windsaloftR = callWinds(wind,routeR)    
            else:
                windsaloftR = [np.zeros(len(routeR.waypoints)),np.zeros(len(routeR.waypoints))]
            
            routeR.deviation = routeR.deviationcheck(parent,optimizationpriority,windsaloftR)
            Route.active[-1] = routeR
            
            # plot right route in green        
    #        rightpltX,rightpltY = zip(*routeR.waypoints)
    #        plt.plot(rightpltX,rightpltY,'--',color='g')  


        # (E) plot alternative waypoints (right waypoints in green, left waypoints in red)
            

    #        for i in range(len(altWptR)):
    #            plt.scatter(altWptR[i][0],altWptR[i][1],color='green')
    #        for i in range(len(altWptL)):
    #            plt.scatter(altWptL[i][0],altWptL[i][1],color='red')
            # import code
            # code.interact(local= locals())
                
        else:
            print("no route intersections!")
            
            # update incumbent if either no incumbent exists
            # or if this route (with no intersections) is shorter than current incumbent
            if incflag == 0:
                incumbent = parent
                print('* first incumbent')
                print('')
                incflag = 1
            elif optimizationpriority == 0:
                if parent.distance < incumbent.distance:
                    incumbent = parent
                    print('--> incumbent updated')
                    print('')
            elif optimizationpriority == 1:
                if parent.time < incumbent.time:
                    incumbent = parent
                    print('--> incumbent updated')
                    print('')

    print('############## end algorithm ##############')
    print('')

    IncX,IncY = list(zip(*incumbent.waypoints))
    
    


    if pltfinalsol:
        plt.plot(IncX,IncY,'-')  

    if optimizationpriority == 0:
        print('optimized route distance [NM]: ', incumbent.distance)
        print('')
    elif optimizationpriority == 1:
        print('optimized route time [hr]: ', incumbent.time)
        print('')
            
    waypoint_latlon = []
    for element in incumbent.waypoints:
        waypoint_latlon.append(XY2LatLon(element[0], element[1]))

    # import code
    # code.interact(local= locals())
    # SHOW INTERACTIVE PLOT WINDOW
    plt.show()
    
    return waypoint_latlon