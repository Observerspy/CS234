class config():
    # env config
    render_train     = True
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/test/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "video/"


    # model and training config
    num_episodes_test = 10
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 1000
    log_freq          = 50
    eval_freq         = 1000
    record_freq       = 1000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 10000
    batch_size         = 32
    buffer_size        = 1000
    target_update_freq = 1000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr                 = 0.0001
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000
    learning_start     = 500
