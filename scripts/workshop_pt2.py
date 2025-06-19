""" 
This is an example on how to use subclassing to modify existing environments.
For this to work, this class needs to be added inside bluesky_gym/envs/sector_cr_env.py
If you base your subclass on another existing environment, 
it should be added to the corresponding file

Additionally, your environment should be added to the register_envs() function
located under bluesky_gym/__init__.py and inside bluesky_gym/envs/__init__.py
"""

class JansEnvironment(SectorCREnv):

    def __init__(self, render_mode=None, ac_density_mode="normal"):
        super().__init__(render_mode=render_mode, ac_density_mode=ac_density_mode)

        # If we were to change something to _get_obs(), change it here aswell!
        self.observation_space = spaces.Dict(
            {
                "cos(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "sin(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "airspeed": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "x_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "y_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vx_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vy_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "cos(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "sin(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "distances": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64)
            }
        )

        # Lets make the action use only speed changes
        # so we update the action_space to be of shape=(1,) 
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
    
    def _get_action(self,action):
        #dh = action[0] * D_HEADING
        dv = action[0] * D_VELOCITY
        #heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(ACTOR)] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx(ACTOR)] + dv) * MpS2Kt

        #bs.stack.stack(f"HDG {ACTOR} {heading_new}")
        bs.stack.stack(f"SPD {ACTOR} {speed_new}")