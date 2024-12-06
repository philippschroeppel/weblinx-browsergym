import os

import gymnasium as gym
from browsergym.core.action.highlevel import HighLevelActionSet
import weblinx_browsergym

os.environ['BROWSERGYM_WEBLINX_REGISTER_TRAIN'] = "False"  
os.environ['BROWSERGYM_WEBLINX_REGISTER_VALID'] = "False"
os.environ['BROWSERGYM_WEBLINX_REGISTER_TEST'] = "True"
os.environ['BROWSERGYM_WEBLINX_REGISTER_TEST_OOD'] = "False"

def test_register():
    # register all tasks on import
    assert len(weblinx_browsergym.ALL_WEBLINX_TASK_IDS) > 0

def test_gym():
    env = gym.make(
        "browsergym/weblinx.arlgkzv.1",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset(seed=123)
    action = 'click(bid="baf79046-bd85-4867")'
    obs, reward, done, truncated, info = env.step(action)

    assert reward == 0.0
    assert done == True
    assert truncated == True
    assert obs is not None
    
def test_load():
    action = 'goto("https://imgur.com/")'
    env = gym.make(
        "browsergym/weblinx.scicrdo.3",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )
    obs, info = env.reset(seed=123)
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0

def test_tabcreate():
    action = 'new_tab()'
    env = gym.make(
        "browsergym/weblinx.qfaggiz.20",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )
    obs, info = env.reset(seed=123)
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0
    assert done == True
    assert truncated == True
    assert obs is not None

def test_submit():
    action = 'click("498ce0d8-f619-42fc")'
    env = gym.make(
        "browsergym/weblinx.ldasoxo.14",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )
    
    obs, info = env.reset(seed=123)
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0
    assert obs is not None

    # trigger submit

    env = gym.make(
        "browsergym/weblinx.ldasoxo.16",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )
    
    obs, info = env.reset(seed=123)

def test_scroll():
    task = "browsergym/weblinx.iszaysr.19"

    env = gym.make(
        task,
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset()
    action = 'scroll(150, 10)'
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0
    assert done == True
    assert truncated == True
    assert obs is not None

    obs, info = env.reset()
    action = 'click(bid="baf79046-bd85-4867")'
    obs, reward, done, truncated, info = env.step(action)

    assert reward == 0
    assert done == True
    assert truncated == True
    assert obs is not None

def test_tabs():
    task = "browsergym/weblinx.qfaggiz.21"

    env = gym.make(
        task,
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset()
    action = 'tab_focus(1482524450)'
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0
    assert done == True
    assert truncated == True
    assert obs is not None


    obs, info = env.reset()
    action = 'tab_close(1482524450)'
    obs, reward, done, truncated, info = env.step(action)

    assert reward == 0
    assert done == True
    assert truncated == True
    assert obs is not None


    obs, info = env.reset()
    action = 'new_tab(1482524450)'
    obs, reward, done, truncated, info = env.step(action)

    assert reward == 0
    assert done == True
    assert truncated == True
    assert obs is not None

def test_fill():
    task = "browsergym/weblinx.qfaggiz.21"

    env = gym.make(
        task,
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset()
    action = 'fill(bid="baf79046-bd85-4867", value="test")'
    obs, reward, done, truncated, info = env.step(action)

    assert reward == 0
    assert done == True
    assert truncated == True
    assert obs is not None


def test_tabremove():
    task = "browsergym/weblinx.qfaggiz.28"

    env = gym.make(
        task,
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset()
    action = 'tab_close()'
    obs, reward, done, truncated, info = env.step(action)

    assert reward > 0
    assert done == True
    assert truncated == True
    assert obs is not None



def test_gym_with_chat():
    env = gym.make(
        "browsergym/weblinx.scicrdo.26",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset(seed=123)

    wenv: weblinx_browsergym.WeblinxEnv = env.env.env
    msgs = wenv.chat.messages

    wenv.chat.add_message("user", "test")
    wenv.chat.wait_for_user_message()
    assert len(msgs) == 1
    

def test_send_msg_to_user():
    env = gym.make(
        "browsergym/weblinx.scicrdo.4",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )

    obs, info = env.reset(seed=123)

    action = 'send_msg_to_user("it is")'

    obs, reward, done, truncated, info = env.step(action)

    assert 1 > reward > 0
    assert obs is not None

def test_hover():
    action = 'hover("cd40c7ea-a9ba-40ef")'
    env = gym.make(
        "browsergym/weblinx.wsonsyq.22",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        wait_for_user_message=False,
        action_mapping=HighLevelActionSet().to_python_code,
    )
    obs, info = env.reset(seed=123)
    obs, reward, done, truncated, info = env.step(action)

    assert 1 > reward > 0
    assert obs is not None
