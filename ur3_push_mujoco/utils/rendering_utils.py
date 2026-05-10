import numpy as np
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer, BaseRender, MujocoRenderer, WindowViewer

class CustomOffScreenViewer(OffScreenViewer):
    def __init__(self, model, data, width, height, geomgroup=None):
        self._get_opengl_backend(width, height)
        BaseRender.__init__(self, model, data, width, height)

        if geomgroup is not None:
            self.set_geomgroup(geomgroup)

    def set_geomgroup(self, geomgroup):
        assert geomgroup.shape == (6,)
        assert geomgroup.dtype == np.uint8
        self.vopt.geomgroup = geomgroup

class CustomMujocoRenderer(MujocoRenderer):
    def __init__(self, model, data, default_cam_config=None):
        super().__init__(model, data, default_cam_config)

    def render(self, render_mode, camera_id=None, camera_name=None, width=None, height=None, geomgroup=None):
        viewer = self._get_viewer(render_mode=render_mode, width=width, height=height, geomgroup=geomgroup)
        if render_mode in {
            "rgb_array",
            "depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

            img = viewer.render(render_mode=render_mode, camera_id=camera_id)
            return img
        elif render_mode == "human":
            return viewer.render()

    def _get_viewer(self, render_mode, width=None, height=None, geomgroup=None):
        if render_mode == "rgb_array" and width is None and height is None:
            viewer_name = "rgb_array_h"
        else:
            viewer_name = render_mode

        self.viewer = self._viewers.get(viewer_name)
        if render_mode == "rgb_array" and width is not None and height is not None and self.viewer is None:
            self.viewer = CustomOffScreenViewer(model=self.model, data=self.data, width=width, height=height, geomgroup=geomgroup)
            self._viewers[viewer_name] = self.viewer
        elif render_mode == "rgb_array" and isinstance(self.viewer, CustomOffScreenViewer) and geomgroup is not None:
            self.viewer.set_geomgroup(geomgroup)
        elif render_mode == "rgb_array" and width is None and height is None and self.viewer is None:
            self.model.vis.global_.offwidth = 1240
            self.model.vis.global_.offheight = 1080
            self.viewer = CustomOffScreenViewer(model=self.model, 
                                                data=self.data, 
                                                width=self.model.vis.global_.offwidth, 
                                                height=self.model.vis.global_.offheight)
            self._set_cam_config()
            self._viewers[viewer_name] = self.viewer
        elif render_mode == "human":
            super()._get_viewer(render_mode)
            
        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer
    
    def reload_model(self, model, data):
        self.model = model
        self.data = data

        for viewer in self._viewers.values():
            viewer.model = model
            viewer.data = data

    