from .main import planner_with_success

def add_planner_args(parser):
    parser.add_argument('--planner', type=str, default='mcts', help='planner to use')
    parser.add_argument('--planner-budget', type=int, default=6000, help='planner budget')
    parser.add_argument('--planner-max-depth', type=int, default=None, help='planner max depth')
    parser.add_argument('--planner-ucb-c', type=float, default=1.0, help='planner ucb c')
def get_planner_args(args):
    if args.planner_max_depth is None:
        if args.env.lower().strip() == 'alfworld':
            args.planner_max_depth = 10
        elif args.env.lower().strip() == 'minigrid':
            args.planner_max_depth = 30
        else:
            args.planner_max_depth = 30
    return {
        'method': args.planner,
        'budget': args.planner_budget,
        'max_depth': args.planner_max_depth,
        'ucb_c': args.planner_ucb_c,
    }

def planner(
    initial_state, mission, world_model,
    method='bfs', budget=100, max_depth=30, ucb_c=1.0,
):
    return planner_with_success(
        initial_state, mission, world_model,
        method=method, budget=budget, max_depth=max_depth, ucb_c=ucb_c,
    )[0]
