def extra_args_provider(parser):
    """difflm arguments."""
    # common
    group = parser.add_argument_group(title='difflm')
    group.add_argument('--debug', action='store_true',
                       help='Whether to run in debug mode.')
    group.add_argument('--turn-on-debugpy', action='store_true',
                       help='Whether to turn on debugpy.')
    group.add_argument('--detect-anomaly', action='store_true',
                       help='Whether to detect anomaly.')
    group.add_argument('--model-running-mode', type=str, default="difflm-noshift", choices=["difflm-noshift", "vanilla", "test-forward"], 
                       help='Running mode, difflm-noshift means vanilla dlm, vanilla means AR.')
    group.add_argument('--base-model', type=str, default="Qwen_2", choices=["vanilla"],
                       help='Use Qwen 2 series model by default (including Qwen 2.5, Qwen 2.5 Math, etc.). This only specifies the model series, instead of the specific model name.')
    group.add_argument('--mask-token', type=int, default=151656, # 151656 is the <|video_pad|> token of Qwen 2.5
                       help='Mask token to use for the diffusion lm.')
    group.add_argument('--attention-mask-type', type=str, default='causal_bottom_right', choices=["causal_bottom_right", "no_mask", "block_causal", "causal"],
                       help='Attention mask type to use for the diffusion lm.')
    group.add_argument('--core-attn-implementation', type=str, default='TEDotProductAttention', choices=["TEDotProductAttention", "FlexAttention"],
                       help='Core attention implementation to use for the diffusion lm.')
    group.add_argument('--non-special-vocab-size', type=int, default=100255,
                       help='The size of the non-special vocab. This is used to replace the model input for uniform tokens.')
    group.add_argument('--cut-off-varlen-to-seqlen', action='store_true',
                       help='Whether to cut off the variable length sequence to the max sequence length. This is only used when we use document packing and intra-doc attention mask.')
    group.add_argument('--checkpoint-steps', type=str, default=None,
                       help='The steps to save the checkpoint, separated by commas.')
    group.add_argument('--difflm-varilen-prob', type=float, default=0.01,
                       help='The probability of using variable length data for the diffusion lm.')
    group.add_argument('--difflm-stable-training', action='store_true',
                       help='When enabled, training loss uses CE * mask (no 1/t weighting) for gradient stability. '
                            'When disabled, training loss uses CE * mask / t (same ELBO weighting as validation).')
    
    # MoE arguments
    group.add_argument('--moe-router-type', type=str, default='expert_choice',
                       choices=['expert_choice', 'topk'],
                       help='MoE router type: expert_choice or topk (token-choice).')
    group.add_argument('--moe-router-dynamic-topk', action='store_true',
                       help='Enable dynamic topk based on mask ratio. Linear scheduler uses ceil(mask_ratio * max_topk).')
    group.add_argument('--moe-router-dynamic-topk-scheduler', type=str, default='linear',
                       choices=['linear', 'cosine', 'linear_reverse', 'cosine_reverse', 'bell', 'bell_gaussian', 'bell_gaussian_reverse'],
                       help='Scheduler type for dynamic topk calculation. linear: topk = ceil(mask_ratio * max_topk). linear_reverse: topk = ceil((1 - mask_ratio) * max_topk). cosine: applies cosine schedule on mask_ratio. cosine_reverse: applies cosine schedule on (1 - mask_ratio). bell: bell-shaped scheduler where topk=1 at mask_ratio=0 and 1, topk=max_topk at mask_ratio=0.5. bell_gaussian: Gaussian bell-shaped scheduler with adjustable sigma parameter. bell_gaussian_reverse: reverse Gaussian bell where topk=max at edges and min at center.')
    group.add_argument('--moe-router-dynamic-topk-max', type=int, default=None,
                       help='Maximum topk value for dynamic topk. Defaults to moe-router-topk value (default 12).')
    group.add_argument('--moe-router-dynamic-topk-min', type=int, default=1,
                       help='Minimum topk value for dynamic topk. Default: 1.')
    group.add_argument('--moe-router-dynamic-topk-bell-sigma', type=float, default=0.2,
                       help='Sigma parameter for bell_gaussian scheduler. Controls the width of the bell curve. Smaller values (e.g., 0.1) create narrower curves, larger values (e.g., 0.3) create wider curves. Default: 0.2.')
    
    # eval arguments
    group.add_argument('--use-on-the-fly-eval', action='store_true',
                       help='Whether to use on-the-fly eval.')
    group.add_argument('--on-the-fly-eval-interval', type=int, default=None,
                       help='Run on-the-fly eval every N steps. Must be a multiple of --eval-interval. '
                            'Default: same as --eval-interval (run at every validation).')
    group.add_argument('--on-the-fly-eval-batch-size', type=int, default=None,
                       help='Batch size for on-the-fly evaluation. Default: 2x micro_batch_size.')
    group.add_argument(
        '--on-the-fly-eval-shard-mode',
        type=str,
        default='none',
        choices=['none', 'expert_dp'],
        help='How to shard on-the-fly eval data across ranks. '
             '"none": legacy behavior (all ranks evaluate full task data). '
             '"expert_dp": shard by expert-data-parallel rank, so ranks in the same EP group '
             'see identical samples while expert-DP replicas evaluate disjoint shards.',
    )
    group.add_argument('--on-the-fly-eval-tasks', type=str,
                       default='mmlu,arc_easy,arc_challenge,piqa,winogrande,copa,hellaswag',
                       help='Comma-separated list of eval tasks.')
    group.add_argument('--on-the-fly-eval-mcq-shots', type=str, default='5',
                       help='Fallback comma-separated few-shot counts for MCQ tasks when not specified in --on-the-fly-eval-mcq-shots-by-task.')
    group.add_argument('--on-the-fly-eval-max-samples', type=int, default=None,
                       help='Global max samples per on-the-fly eval task for quick debugging. '
                            'Task-specific sample args (e.g., mmlu/hellaswag) override this when set.')
    group.add_argument(
        '--on-the-fly-eval-mcq-shots-by-task',
        type=str,
        default='mmlu:5,arc_easy:5,arc_challenge:25,piqa:0,winogrande:5,copa:0,boolq:0,openbookqa:0',
        help='Per-task MCQ shots. Format: "task:shots,task:shots", where shots is a single int or pipe-separated list (e.g. mmlu:0|5). '
             'Supported tasks: mmlu, arc_easy, arc_challenge, piqa, winogrande, copa, boolq, openbookqa. '
             'Set to empty string to disable per-task overrides and use --on-the-fly-eval-mcq-shots for all MCQ tasks.',
    )
    group.add_argument('--on-the-fly-eval-mmlu-samples', type=int, default=None,
                       help='Number of MMLU samples to evaluate. None = all test samples.')
    group.add_argument('--on-the-fly-eval-hellaswag-samples', type=int, default=None,
                       help='Number of HellaSwag samples to evaluate. None = all validation (~10042).')
    group.add_argument('--on-the-fly-eval-data-dir', type=str, default=None,
                       help='Local data directory for eval datasets. If None, downloads from HuggingFace.')
    group.add_argument(
        '--eval-before-train',
        action='store_true',
        help='Run one validation (and on-the-fly eval if enabled) before entering the training loop.',
    )
    group.add_argument(
        '--reset-dataloader',
        action='store_true',
        help='Reset consumed_train_samples to 0 when resuming from a checkpoint. '
             'Unlike --finetune, this preserves optimizer and LR scheduler state. '
             'Use when changing the data blend mid-training.',
    )
    return parser
