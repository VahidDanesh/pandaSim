import genesis as gs

gs.init()

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        # don't set run_in_thread → on Linux defaults to True,
        # which means the pyrender loop only starts when you call `.run()`
        run_in_thread=True,
        camera_pos   =(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov   =40,
        max_FPS      =60,
    ),
)

scene.build()


