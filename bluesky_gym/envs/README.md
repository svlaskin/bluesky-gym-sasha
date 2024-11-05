# bluesky_gym/envs

This folder contains the implementation of the different environments

## The environments

This document highlights the various environments currently available as benchmarks in BlueSky-Gym. For more information on the specific implementation of the environments, please refer to the source code and accompanying documentation / comments. 

### Vertical Control

Currently there are two vertical control environments: "DescentEnv-v0" and "VerticalCREnv-v0".
The goal of these environments is to stay at the target altitude for as long as possible before initiating the descent towards the runway. The agent controls the vertical velocity of the aircraft. 

In "VerticalCREnv-v0" additional aircraft are operating in the airspace, which are initialized to have a horizontal (so not vertical) conflict with the aircraft, which need to be resolved through vertical velocity changes. If the other aircraft turn red it indicates that the aircraft are in the same horizontal plane, and thus need to be vertically seperated. If the other aircraft is white the aircraft are allowed to be at the same altitude. 

DescentEnv-v0 | VerticalCREnv-v0
:--------------------------------------------------:|:--------------------------------------------------:
<img src="https://github.com/user-attachments/assets/40c47358-65b1-478f-8458-e64a30c86e57" width=50% height=50%>             |<img src="https://github.com/user-attachments/assets/9253d208-539b-4ff1-a7de-335804ab6cbd" width=50% height=50%> 


### Horizontal Conflict Resolution

For horizontal conflict resolution, three different environments have been created: "HorizontalCREnv-v0", "SectorCREnv-v0" and "MergeEnv-v0". The differences between the three environments are highlighted below.

"**HorizontalCREnv-v0**": Conflicts are generated through the Bluesky command 'creconfs', ensuring that the initial state of the other aircraft are generated such that they are all in conflict with the aircraft controlled by the agent. This implies that if the aircraft does not change its path it will experience a loss of separation with all of the aircraft in the environment.
This environment can be used to isolate conflict scenarios and investigate the efficacy of a resolution method.

"**SectorCREnv-v0**": Other aircraft are generated randomly, both in initial state and number. Allowing for better studying of secondary conflict effects as manoeuvres executed by the agent controlled aircraft can lead to new conflict situations. Additionally the varrying number of aircraft in the environment imposes an additional challenge in the state vector construction, which can be studied in this environment. 

"**MergeEnv-v0**": Similar to "SectorCREnv-v0", but focuses on merging scenarios. The agent therefore has to learn to strategically position itself based on the flow of traffic to avoid accumulating a large penalty. 

For all of the aforementioned environments, the agent controls both the heading and speed of the aircraft, and the reward function is composed of both number of intrusions and track deviation from the optimal path to the destination.

HorizontalCREnv-v0 | SectorCREnv-v0 | MergeEnv-v0
:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:
<img src="https://github.com/user-attachments/assets/d61f25a3-8bd8-4b71-9be4-f21b06c35a07" width=75% height=75%>             |<img src="https://github.com/user-attachments/assets/1fd213d1-f11b-40fb-8460-0e7ef52c754e" width=75% height=75%> |<img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width=75% height=75%> 

### Horizontal Control

The horizontal control environments are a set of environments characterized by changes in horizontal velocity and heading, similar to the horizontal CR environments. The purpose of these environments however are not conflict resolution, but other tasks. They currently consist of the "StaticObstacleEnv-v0" and "PlanWaypointEnv-v0".

"**StaticObstacleEnv-v0**": The agent has to avoid static obstacles in the environment to reach a destination waypoint. Unlike the CR environments, where the obstacles are dynamic, this can lead to deadlocks and requires more planning from the agent, instead of reactive control. Currently the state of the obstacles is encoded as the smallest circle capable of enclosing the polygon, research directions can include investigating better ways of encoding the static obstacles in the state space of the agent.

"**PlanWaypointEnv-v0**": The agent has to plan a route, visiting all the waypoints in the environment, similar to a continuous space traveling salesman problem. The main challenge of this environment is determining the priority of the waypoints and dealing with conflicting goals, as can be observed in the trained policy, which oscillates its heading between the different waypoints. 

StaticObstacleEnv-v0 | PlanWaypointEnv-v0
:--------------------------------------------------:|:--------------------------------------------------:
<img src="https://github.com/user-attachments/assets/85a486bd-47bf-4afb-8267-63eebde70407" width=50% height=50%>             |<img src="https://github.com/user-attachments/assets/6dc1574f-0332-4837-a6e4-5d0596512e01" width=50% height=50%> 
