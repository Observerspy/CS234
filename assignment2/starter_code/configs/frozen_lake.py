class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    RGB              = True
    overwrite_render = True

    # output config
    output_path  = "results/test/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    training_path = "results/train/"

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 500
    log_freq          = 50
    eval_freq         = 50000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 2000*200
    batch_size         = 32
    buffer_size        = 50000
    target_update_freq = 5000
    gamma              = 0.99
    learning_freq      = 1
    state_history      = 1
    skip_frame         = 1
    lr                 = 0.1
    eps_begin          = 0.1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train
    learning_start     = 5000
