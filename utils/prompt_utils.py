import textwrap
from typing import List, Dict, Any, Iterator,Tuple
from string import Template
from PIL import Image

SYSTEM_PROMPT_TEMPLATE= textwrap.dedent("""\
You are a visual navigation agent tasked with finding "$goal_name" in an unknown environment.
You will receive a sequence of observations showing your movement history up to the current moment.

**Action Space:**
$action_space_str

**Your Mission:**
1. Analyze the observation history to understand your current location and orientation.
2. Select the next discrete action to navigate efficiently towards the goal.

**Critical Constraints:**
* **Collision Detection:** If your previous action was **forward** but the visual observation did not change significantly, you have collided. You MUST turn or move away immediately. Do not keep pushing forward.
* **Success Condition:** Output **stop** ONLY when the target is plainly in view, centered, and within 1 meter (close enough to touch).

**Output Format:**
Respond with the selected action inside double asterisks.
""")


CONVO_TURN_TEMPLATE = [
    {
        "role": "assistant",
        "content":[
            {"type":"text","text": "**$action**"} # tell the agent what its last action was with substitution
        ]
    },
    {
        "role": "user",
        "content": [ # Placeholder for the pixel data
            {"type": "image"}
            # {"type": "image", "text": "Observation {$step} at {$agent_loc} heading {$agent_heading}"}
        ],
    },
    {
        "role": "assistant",
        "content":[
            {"type":"text","text": "**forward**"} # placeholder action to infer logprob during forward pass
        ]
    }
]

CONVO_START_TEMPLATE = [
    {
        "role": "user",
        "content": [ # Placeholder for the pixel data
            {"type": "text", "text": SYSTEM_PROMPT_TEMPLATE},
        ],
    },
    {
        "role": "user",
        "content": [ # Placeholder for the pixel data
            {"type": "image"}
            # {"type": "image", "text": "Observation {$step} at {$agent_loc} heading {$agent_heading}"}
        ],
    },
    {
        "role": "assistant",
        "content":[
            {"type":"text","text": "**forward**"} # placeholder action to infer logprob during forward pass
        ]
    }
]


def substitute_convo_template(conversation_template: List[Dict], substitutions: Dict[str, Any]) -> List[Dict]:
    """
    Traverses the conversation template and substitutes any string.Template 
    objects found in 'text' fields using values from the 'obs' dictionary.
    
    Args:
        conversation_template: List of message dicts (role, content).
        substitutions: Dictionary containing substitution keys (e.g., 'goal_name').
        
    Returns:
        A new conversation list with strings substituted.
    """
    new_conversation = []
    
    for message in conversation_template:
        # Shallow copy the message container
        new_message = message.copy()
        new_content = []
        
        # Iterate over the content list (e.g., [{"type": "image"}, {"type": "text", ...}])
        for item in message.get("content", []):
            new_item = item.copy()
            
            # Check if this item is a text component
            if "text" in new_item:
                text_obj = new_item["text"]
                
                # CASE A: It's a Template object (from the config)
                if "$" in text_obj:
                    try:
                        text_template = Template(text_obj)
                        # Perform the substitution
                        new_item["text"] = text_template.substitute(substitutions)
                    except KeyError as e:
                        raise
                        # Fallback to safe_substitute to prevent crashing on missing keys,
                        # but log it so we know something is wrong.
                        # print(f"Warning: Missing substitution key {e} in template.")
                        # new_item["text"] = text_template.safe_substitute(substitutions)
                        
                # CASE B: It's already a str (static text)
                elif isinstance(text_obj, str):
                    pass # Keep as is
                    
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
        
    return new_conversation

def prepare_turn_inputs(step_idx, action):
    # Construct a standalone prompt for this step
    # We simulate the user asking for a move
    if step_idx>0:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ]
            },
            { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": f"**{action}**"}]
                }]
    else:
        messages = [
                {
                    "role": "user",
                    "content": [ # Placeholder for the pixel data
                        {"type": "text", "text": SYSTEM_PROMPT_TEMPLATE},
                    ],
                },
                {
                    "role": "user",
                    "content": [ # Placeholder for the pixel data
                        {"type": "image"}
                    ],
                },
                # We add the Assistant start token to force the model to predict the response immediately
                { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": f"**{action}**"}]
                }]
    return messages

# action_mapping = {'STOP': 'stop', 'MOVE_FORWARD': 'forward', 
#                         'TURN_LEFT': 'left', 'TURN_RIGHT': 'right', 'LOOK_UP': 'up', 'LOOK_DOWN': 'down'}

def prepare_sequence_inputs(examples, info):
    messages = []
    actions = examples['action_sequence']
    images = examples['images']
    for i in range(len(images)-1):
        # message = prepare_turn_inputs(i, action_mapping[actions[i]])
        message = prepare_turn_inputs(i, actions[i])
        message = substitute_convo_template(message, info)
        messages += message
    messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                ]
            },
            {
                "role": "assistant",
                "content":[
                    {"type":"text","text": "**"} # placeholder action to infer logprob during forward pass
                ]
            }]
    return messages, images
