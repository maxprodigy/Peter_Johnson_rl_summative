import sys
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_HELVETICA_12

def norm(c): return [x / 255.0 for x in c]
WHITE = norm((255, 255, 255))
BLACK = norm((0, 0, 0))
GREEN = norm((144, 238, 144))
BLUE = norm((100, 149, 237))
RED = norm((255, 99, 71))
YELLOW = norm((255, 255, 0))
GREY = norm((200, 200, 200))

def visualize_grid_env(env, episodes=3, fps=5, tile_size=100, model=None):
    # Get grid shape
    GRID_HEIGHT = getattr(env, 'grid_size', getattr(env, 'grid_height', 5))
    GRID_WIDTH = getattr(env, 'grid_size', getattr(env, 'grid_width', 5))
    WIDTH = tile_size * GRID_WIDTH
    HEIGHT = tile_size * GRID_HEIGHT + 50  # extra space for info bar

    # Get label dict or empty
    TILE_TYPES = getattr(env, 'TILE_TYPES', {})
    if not TILE_TYPES:
        TILE_TYPES = getattr(env, 'tile_types', {})

    step_count = 0
    episode_count = 1
    done = False
    reward = 0

    obs, _ = env.reset()

    def draw_text(x, y, text, font=GLUT_BITMAP_HELVETICA_18, color=BLACK):
        glColor3fv(color)
        glRasterPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(font, ord(ch))

    def draw_grid():
        for row in range(GRID_HEIGHT):
         for col in range(GRID_WIDTH):
            x = col * tile_size
            y = row * tile_size
            tile_color = GREEN
            if hasattr(env, "goal") and (row, col) == env.goal:
                tile_color = RED
            elif hasattr(env, "obstacles") and (row, col) in env.obstacles:
                tile_color = BLUE
            glColor3fv(tile_color)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + tile_size, y)
            glVertex2f(x + tile_size, y + tile_size)
            glVertex2f(x, y + tile_size)
            glEnd()
            # Border
            glColor3fv(BLACK)
            glLineWidth(1)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x, y)
            glVertex2f(x + tile_size, y)
            glVertex2f(x + tile_size, y + tile_size)
            glVertex2f(x, y + tile_size)
            glEnd()
            # Centered Label if present
            if (row, col) in TILE_TYPES:
                label = TILE_TYPES[(row, col)]
                label_width = len(label) * 9
                label_x = x + (tile_size - label_width) // 2
                label_y = y + tile_size // 2 + 6
                draw_text(label_x, label_y, label, GLUT_BITMAP_HELVETICA_18, BLACK)


    def draw_agent():
        if hasattr(env, "agent_pos"):
            ax = env.agent_pos[1] * tile_size
            ay = env.agent_pos[0] * tile_size
            glColor3fv(YELLOW)
            glBegin(GL_QUADS)
            glVertex2f(ax, ay)
            glVertex2f(ax + tile_size, ay)
            glVertex2f(ax + tile_size, ay + tile_size)
            glVertex2f(ax, ay + tile_size)
            glEnd()
            # Draw "Creative" label centered
            draw_text(ax + tile_size // 2 - (len("Creative") * 6), ay + tile_size // 2 + 6, "Creative", GLUT_BITMAP_HELVETICA_18, BLACK)

    def draw_info_bar():
        # White bar at bottom
        glColor3fv(WHITE)
        glBegin(GL_QUADS)
        glVertex2f(0, HEIGHT - 50)
        glVertex2f(WIDTH, HEIGHT - 50)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()
        draw_text(10, HEIGHT - 20, f"Step: {step_count}   Reward: {reward}", GLUT_BITMAP_HELVETICA_18, BLACK)

    def display():
        glClear(GL_COLOR_BUFFER_BIT)
        draw_grid()
        draw_agent()
        draw_info_bar()
        glutSwapBuffers()

    def update(_):
        nonlocal obs, step_count, episode_count, done, reward
        if done:
            if episode_count < episodes:
                episode_count += 1
                obs, _ = env.reset()
                step_count = 0
                done = False
                reward = 0
                glutPostRedisplay()
                glutTimerFunc(1000, update, 0)
            else:
                print("Visualization done!")
                time.sleep(1)
                try:
                    glutDestroyWindow(glutGetWindow())
                except Exception as e:
                    print("Window close error:", e)
                sys.exit(0)
        else:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
            glutPostRedisplay()
            glutTimerFunc(int(1000 / fps), update, 0)

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"Grid Env Visualization")
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, HEIGHT, 0)
    glMatrixMode(GL_MODELVIEW)
    glutDisplayFunc(display)
    glutTimerFunc(int(1000 / fps), update, 0)
    glutMainLoop()