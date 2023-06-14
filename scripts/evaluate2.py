import q_learning as ql
import pickle

def eval_loop(algo):

    gapsizes = [0.012, 0.015, 0.02, 0.04]

    results = {}
    for g in gapsizes:
        results['gapsize' + str(g)] = [] 
        for _ in range(10):
            algo.gapsize = g
            _, actions, r, t = algo.test(test_action=None, evaluate=True)

            results['gapsize' + str(g)].append((r, t, actions))
    
    return results

def run_auto_eval():


    filepath = '/home/marco/Masterarbeit/Training/AllRuns/20-05-2023-16:05'
    
    
    algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=3, n_timesteps=10, pitch_only=True)
    
    algo.max_ep_steps = 100
    algo.angle_range = 0.08
    # results = eval_loop(algo)
      
    # with open(filepath + '/resultsEZ.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/05-06-2023-09:19'
    
    
    # algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=3, n_timesteps=20, pitch_only=True)
    
    # algo.max_ep_steps = 150
    # results = eval_loop(algo)
    
    # with open(filepath + '/results.pickle', 'wb') as f:
    #     pickle.dump(results, f)


    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/03-06-2023-23:45'
    
    
    # algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=1, n_timesteps=10, pitch_only=True)
    
    # algo.max_ep_steps = 150
    # results = eval_loop(algo)


    # with open(filepath + '/results.pickle', 'wb') as f:
    #     pickle.dump(results, f)
    
    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/04-06-2023-15:51'
    
    
    # algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=0, n_timesteps=10, pitch_only=True)
    
    # algo.max_ep_steps = 150
    # results = eval_loop(algo)


    # with open(filepath + '/results.pickle', 'wb') as f:
    #     pickle.dump(results, f)
    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/08-06-2023-10:39'
    
    
    # algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=2, n_timesteps=10, pitch_only=True)
    
    # algo.max_ep_steps = 150
    # results = eval_loop(algo)


    # with open(filepath + '/results.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    # algo.env.close
    
    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/27-05-2023-08:42'
    
    
    # algo = ql.DQN_Algo(filepath=filepath, sensor="plate", eval=True, actionspace='full', reduced=2, n_timesteps=10, pitch_only=False)
    
    # algo.max_ep_steps = 150
    # results = eval_loop(algo)


    # with open(filepath + '/results.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    # algo.env.close()
     
    
if __name__=='__main__':

    filepath = '/home/marco/Masterarbeit/Training/AllRuns/01-06-2023-10:40'
    
    # filepath = '/home/marco/Masterarbeit/Training/AllRuns/03-06-2023-23:45'
    algo = ql.DQN_Algo(filepath=filepath, sensor="fingertip", eval=True, actionspace='full', reduced=3, n_timesteps=10, pitch_only=True)
    
    algo.max_ep_steps = 90
    # algo.angle_range = 0.04
    # results = eval_loop(algo)


    # with open(filepath + '/results_EZ.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    # run_auto_eval()
    # gapsize = [0.002, 0.002, 0.002, 0.002]
    # algo.angle_range = 0.08
    # algo.sim_steps = 10
    
    # algo.max_ep_steps = 150
    algo.gapsize = 0.012
    r = 0
    # while r < 0.9:
    (q, tactile, r, t), actions, r, _  = algo.evaluate(test_action=None)    
    
    states = (tactile, r, t, actions)

    with open('/home/marco/Masterarbeit/ModelEvaluation/ACTIONEVAL_REDUCED3-.pickle', 'wb') as f:
        pickle.dump((q, states), f)

    # algo.gapsize = 0.02
    # q, states = algo.test(test_action=0, evaluate=True)    
    
    # # with open('/home/marco/Masterarbeit/ModelEvaluation/no_contact.pickle', 'wb') as f:
    # #     pickle.dump((q, states), f)
