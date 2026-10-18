"""Watch your trained AI drive, or race against it.

    python watch.py                              watch the best model of run1
    python watch.py --model models/numpy1_best.npz
    python watch.py --race                       you (arrows) vs the AI
    python watch.py --duel models/run1_best.pt models/numpy1_best.npz     AI vs AI
    add  --track hairpin  to any of these to use another track (T in watch mode: next track)
"""
import argparse
import os
import sys
import pygame
import config
from env.racing_env import RacingEnv
from agent.dqn_agent import DQNAgent
from game.race import Race
from game import race as drawing
from game.tracks import track_names
from agent.brains import saved_input_size, lookahead_for


def load_agent(model_path, env):
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}\nTrain one first with:  python train.py")
    backend = "numpy" if model_path.endswith(".npz") else "torch"
    n_inputs = saved_input_size(model_path)
    env.set_lookahead(lookahead_for(n_inputs))         # old models: 9 inputs, new: 11
    agent = DQNAgent(n_inputs, env.n_actions, backend=backend, buffer_size=1)
    agent.load(model_path)
    agent.lookahead = env.lookahead
    print(f"Loaded {model_path} ({backend} network, {n_inputs} inputs)")
    return agent

def setup_window(title):
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(title)
    fonts = (pygame.font.Font(None, 30), pygame.font.Font(None, 48), pygame.font.Font(None, 22))
    return screen, pygame.time.Clock(), fonts


def draw_q_panel(screen, font, q_values, chosen, center):
    """Five bars: how good the AI thinks each action is right now."""
    cx, cy = center
    left, top, bar_width = cx - 160, cy - 45, 160
    box = pygame.Surface((340, 96), pygame.SRCALPHA)
    box.fill((0, 0, 0, 110))                                         # see-through dark box
    screen.blit(box, (left - 8, top - 4))
    low, high = float(min(q_values)), float(max(q_values))
    span = max(high - low, 1e-6)
    for i, (name, q) in enumerate(zip(RacingEnv.ACTION_NAMES, q_values)):
        y = top + i * 18
        colour = config.COLOR_Q_CHOSEN if i == chosen else config.COLOR_Q_BAR
        length = int(10 + (q - low) / span * (bar_width - 10))      # scaled so differences show
        label = font.render(f"{name:>13}", True, config.COLOR_HUD_TEXT)
        screen.blit(label, (left, y))
        pygame.draw.rect(screen, colour, (left + 110, y + 2, length, 12))
        value = font.render(f"{q:6.2f}", True, colour)
        screen.blit(value, (left + 110 + bar_width + 8, y))


# =====================================================================
#  Watch mode
# =====================================================================

def watch(model_path, track):
    all_tracks = track_names()
    env = RacingEnv(tracks=all_tracks)                 # every track built once, so T can switch instantly
    agent = load_agent(model_path, env)
    screen, clock, (font, big_font, small_font) = setup_window(f"AI Racer - Watching the AI - {track}")
    sprite = drawing.CarSprite(config.CAR_SPRITE_PATH, config.CAR_HUE_SHIFT)

    def start(name):
        pygame.display.set_caption(f"AI Racer - Watching the AI - {name}")
        print(f"--- {name} ---")
        return env.reset(track=name), drawing.build_background(env.races[name].track)

    observation, background = start(track)
    q_values = agent.q_values(observation)
    action = int(q_values.argmax())
    paused, fast = False, False
    message, message_timer = "", 0
    laps_seen = 0
    finished_timer = 0                     # pause after an episode ends, before restarting
    controls = "Space: pause   F: fast   R: restart   T: next track   Esc: quit"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    fast = not fast
                elif event.key == pygame.K_r:
                    observation, laps_seen, finished_timer = env.reset(track=track), 0, 0
                elif event.key == pygame.K_t:
                    track = all_tracks[(all_tracks.index(track) + 1) % len(all_tracks)]
                    (observation, background), laps_seen, finished_timer = start(track), 0, 0

        if not paused and finished_timer == 0:
            for _ in range(4 if fast else 1):
                q_values = agent.q_values(observation)
                action = int(q_values.argmax())
                observation, reward, terminated, truncated, info = env.step(action)

                if info["laps"] > laps_seen:                      # a lap was just completed
                    laps_seen = info["laps"]
                    lap = env.race.last_lap_steps / config.STEPS_PER_SECOND
                    record = config.HUMAN_RECORDS.get(track)
                    if record:
                        difference = lap - record
                        verdict = "faster" if difference < 0 else "slower"
                        print(f"Lap {laps_seen}: {lap:.2f} s  ({abs(difference):.2f} s {verdict} than your {record} s)")
                    else:
                        print(f"Lap {laps_seen}: {lap:.2f} s  (no record of yours on {track} yet)")
                    message, message_timer = f"LAP {lap:.2f} s", 30

                if terminated or truncated:
                    print(f"Episode over: {info['reason']}  (reward {info['episode_reward']:.1f})\n")
                    message, message_timer = info["reason"].upper(), 30
                    finished_timer = 30                            # 2 seconds at 15 fps
                    break
        elif finished_timer > 0:
            finished_timer -= 1
            if finished_timer == 0:
                observation, laps_seen = env.reset(track=track), 0

        if message_timer > 0:
            message_timer -= 1
        else:
            message = ""

        car = env.race.car
        screen.blit(background, (0, 0))
        drawing.draw_rays(screen, (car.x, car.y), env.ray_hit_points)
        sprite.draw(screen, car, False)
        drawing.draw_hud(screen, env.race, font, big_font, small_font, "", controls)
        draw_q_panel(screen, small_font, q_values, action, env.race.track.q_panel_center)
        shown = ("PAUSED" if paused else message) + ("   (x4)" if fast else "")
        if shown:                                   # drawn just above the car, clear of the panels
            x = min(max(car.x, 130), config.SCREEN_WIDTH - 130)
            y = max(car.y - 45, 30)
            drawing.draw_text_centered(screen, big_font, shown, config.COLOR_MESSAGE, (x, y))
        pygame.display.flip()
        clock.tick(config.FPS // config.FRAME_SKIP)              # real-time speed

    pygame.quit()


# =====================================================================
#  Race mode: you vs the AI
# =====================================================================

def race(model_path, laps, track):
    ai_env = RacingEnv(wall_mode="solid", track_name=track)    # in a race, walls block both cars
    agent = load_agent(model_path, ai_env)
    screen, clock, (font, big_font, small_font) = setup_window("AI Racer - You vs the AI")
    background = drawing.build_background(ai_env.race.track)
    human_sprite = drawing.CarSprite(config.CAR_SPRITE_PATH, config.CAR_HUE_SHIFT)
    ai_sprite = drawing.CarSprite(config.CAR_SPRITE_PATH, config.AI_CAR_HUE_SHIFT)
    human = Race(wall_mode="solid", allow_reverse=True, track_name=track)
    cx, cy = human.track.hud_center

    def new_race():
        human.reset()
        ai_env.reset()
        return {"frame": 0, "countdown": 3 * config.FPS, "action": 0,
                "human_total": None, "ai_total": None, "human_flash": 0, "winner": None}

    state = new_race()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = new_race()

        if state["countdown"] > 0:
            state["countdown"] -= 1
        else:
            state["frame"] += 1

            # --- you ---
            if state["human_total"] is None:
                keys = pygame.key.get_pressed()
                steer = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
                events = human.step(keys[pygame.K_UP], keys[pygame.K_DOWN], steer)
                if events["crashed"]:
                    state["human_flash"] = 10
                if human.laps_completed >= laps:
                    state["human_total"] = state["frame"]

            # --- the AI: decides every FRAME_SKIP frames, exactly like in training ---
            if state["ai_total"] is None:
                if state["frame"] % config.FRAME_SKIP == 1:
                    state["action"] = agent.act(ai_env._get_observation(), explore=False)
                throttle, brake, ai_steer = RacingEnv.ACTIONS[state["action"]]
                ai_env.race.step(throttle, brake, ai_steer)
                if ai_env.race.laps_completed >= laps:
                    state["ai_total"] = state["frame"]

            if state["winner"] is None:
                if state["human_total"] is not None:
                    state["winner"] = "YOU WIN!"
                elif state["ai_total"] is not None:
                    state["winner"] = "THE AI WINS!"

        if state["human_flash"] > 0:
            state["human_flash"] -= 1

        # --- draw ---
        screen.blit(background, (0, 0))
        ai_sprite.draw(screen, ai_env.race.car, False)
        human_sprite.draw(screen, human.car, state["human_flash"] > 0)

        def car_line(name, r, total):
            if total is not None:
                return f"{name}: FINISHED in {total / config.STEPS_PER_SECOND:.2f} s"
            best = f"{r.best_lap_steps / config.STEPS_PER_SECOND:.2f} s" if r.best_lap_steps else "--"
            return f"{name}: lap {min(r.laps_completed + 1, laps)}/{laps}   best lap {best}"

        drawing.draw_text_centered(screen, font, car_line("YOU (red)", human, state["human_total"]),
                                   config.COLOR_HUD_TEXT, (cx, cy - 60))
        drawing.draw_text_centered(screen, font, car_line("AI (blue)", ai_env.race, state["ai_total"]),
                                   config.COLOR_HUD_TEXT, (cx, cy - 25))
        drawing.draw_text_centered(screen, small_font, f"Race time: {state['frame'] / config.STEPS_PER_SECOND:.2f} s",
                                   config.COLOR_HUD_TEXT, (cx, cy + 10))

        if state["countdown"] > 0:
            number = state["countdown"] // config.FPS + 1
            drawing.draw_text_centered(screen, big_font, str(number), config.COLOR_MESSAGE, (cx, cy + 70))
        elif state["frame"] < config.FPS:
            drawing.draw_text_centered(screen, big_font, "GO!", config.COLOR_MESSAGE, (cx, cy + 70))
        elif state["winner"]:
            drawing.draw_text_centered(screen, big_font, state["winner"], config.COLOR_MESSAGE, (cx, cy + 70))

        drawing.draw_text_centered(screen, small_font,
                                   f"First to {laps} laps wins   Arrows: drive   R: restart race   Esc: quit",
                                   config.COLOR_HUD_TEXT, (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 20))
        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()


# =====================================================================
#  Duel mode: AI vs AI
# =====================================================================

def pick_colours(model_a, model_b, chosen):
    """Colour per car: what you asked for, otherwise by network type (gold = your NumPy network)."""
    if chosen:
        return chosen
    colours = [config.NUMPY_CAR_COLOUR if path.endswith(".npz") else config.TORCH_CAR_COLOUR
               for path in (model_a, model_b)]
    if colours[0] == colours[1]:                       # same type: make the second car blue
        colours[1] = "blue" if colours[0] != "blue" else "green"
    return colours


def duel(model_a, model_b, laps, track, chosen_colours=None):
    names = [os.path.splitext(os.path.basename(path))[0] for path in (model_a, model_b)]
    envs = [RacingEnv(wall_mode="solid", track_name=track), RacingEnv(wall_mode="solid", track_name=track)]
    agents = [load_agent(model_a, envs[0]), load_agent(model_b, envs[1])]
    screen, clock, (font, big_font, small_font) = setup_window("AI Racer - AI vs AI")
    background = drawing.build_background(envs[0].race.track)
    colours = pick_colours(model_a, model_b, chosen_colours)
    sprites = [drawing.CarSprite(config.CAR_SPRITE_PATH, config.CAR_COLOURS[c]) for c in colours]
    cx, cy = envs[0].race.track.hud_center

    def new_duel(random_start):
        gate, nudge = envs[0].random_start_position() if random_start else (0, None)
        for env in envs:
            env.reset(gate, nudge)                     # both cars get the SAME start
        return {"frame": 0, "countdown": 2 * config.FPS, "actions": [0, 0],
                "totals": [None, None], "winner": None, "gate": gate}

    state = new_duel(random_start=False)
    fast = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = new_duel(random_start=False)
                elif event.key == pygame.K_n:
                    state = new_duel(random_start=True)
                elif event.key == pygame.K_f:
                    fast = not fast

        for _ in range(4 if fast else 1):
            if state["countdown"] > 0:
                state["countdown"] -= 1
                continue
            state["frame"] += 1
            for i in range(2):
                if state["totals"][i] is not None:
                    continue                                   # this car has finished
                if state["frame"] % config.FRAME_SKIP == 1:
                    state["actions"][i] = agents[i].act(envs[i]._get_observation(), explore=False)
                envs[i].race.step(*RacingEnv.ACTIONS[state["actions"][i]])
                if envs[i].race.laps_completed >= laps:
                    state["totals"][i] = state["frame"]
                    seconds = state["frame"] / config.STEPS_PER_SECOND
                    if state["winner"] is None:
                        state["winner"] = f"{names[i]} ({colours[i]}) WINS!"
                        print(f"Start gate {state['gate']}:  {names[i]} wins in {seconds:.2f} s")
                    else:
                        print(f"               {names[i]} finishes in {seconds:.2f} s")

        screen.blit(background, (0, 0))
        for i in (1, 0):                                       # first car drawn on top
            sprites[i].draw(screen, envs[i].race.car, False)

        for i in range(2):
            r = envs[i].race
            if state["totals"][i] is not None:
                text = f"{names[i]} ({colours[i]}): FINISHED in {state['totals'][i] / config.STEPS_PER_SECOND:.2f} s"
            else:
                best = f"{r.best_lap_steps / config.STEPS_PER_SECOND:.2f} s" if r.best_lap_steps else "--"
                text = f"{names[i]} ({colours[i]}): lap {min(r.laps_completed + 1, laps)}/{laps}   best lap {best}"
            drawing.draw_text_centered(screen, font, text, config.COLOR_HUD_TEXT, (cx, cy - 60 + i * 35))
        drawing.draw_text_centered(screen, small_font, f"Race time: {state['frame'] / config.STEPS_PER_SECOND:.2f} s"
                                   f"   (start/finish: gate {state['gate']})", config.COLOR_HUD_TEXT, (cx, cy + 20))
        if state["countdown"] > 0:
            drawing.draw_text_centered(screen, big_font, str(state["countdown"] // config.FPS + 1),
                                       config.COLOR_MESSAGE, (cx, cy + 70))
        elif state["winner"]:
            drawing.draw_text_centered(screen, big_font, state["winner"], config.COLOR_MESSAGE, (cx, cy + 70))
        drawing.draw_text_centered(screen, small_font, "F: fast-forward   R: replay   N: new random start   Esc: quit",
                                   config.COLOR_HUD_TEXT, (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 20))
        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch or race your trained AI.")
    parser.add_argument("--model", default=os.path.join("models", "run1_best.pt"))
    parser.add_argument("--race", action="store_true", help="race against the AI")
    parser.add_argument("--duel", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="race two AI models")
    parser.add_argument("--colours", nargs=2, choices=list(config.CAR_COLOURS), metavar=("COLOUR_A", "COLOUR_B"),
                        help=f"duel car colours, from: {', '.join(config.CAR_COLOURS)}")
    parser.add_argument("--laps", type=int, default=config.EPISODE_LAPS)
    parser.add_argument("--track", default=config.DEFAULT_TRACK, choices=track_names())
    args = parser.parse_args()
    if args.duel:
        duel(args.duel[0], args.duel[1], args.laps, args.track, args.colours)
    elif args.race:
        race(args.model, args.laps, args.track)
    else:
        watch(args.model, args.track)