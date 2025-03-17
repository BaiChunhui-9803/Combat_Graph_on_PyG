


def analyze_tactic(s, a):
    # in_cd = -
    pass

def analyze_action(sa_log, action_dict, state_dict):
    state_action_dict = {}
    tactic_dict = {}

    for counter in sa_log:
        for key, value in counter.items():
            if key in state_action_dict:
                state_action_dict[key]['count'] += value
            else:
                state = state_dict[key[0]]
                action = action_dict[key[1]]
                state_action_dict[key] = {
                    'state': state,
                    'action': action,
                    'count': value
                }
                # tactic = analyze_tactic(state, action)
                tactic_dict[key] = []
    pass

def mark_action_label(actions):
    """
    Mark the action with the given label.
    """
    pass
    # action.label = label
    # action.save()
    # return action