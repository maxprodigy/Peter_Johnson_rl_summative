import sys
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_HELVETICA_12
import numpy as np

def norm(c): return [x / 255.0 for x in c]

# Color palette
WHITE = norm((255, 255, 255))
BLACK = norm((0, 0, 0))
GREEN = norm((144, 238, 144))
LIGHT_GREEN = norm((200, 255, 200))
BLUE = norm((100, 149, 237))
RED = norm((255, 99, 71))
YELLOW = norm((255, 215, 0))  
GREY = norm((200, 200, 200))
DARK_GREY = norm((128, 128, 128))

def draw_circle(center_x, center_y, radius, segments=36):
    """Draw a smooth circle with specified segments"""
    glBegin(GL_POLYGON)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        glVertex2f(x, y)
    glEnd()

def draw_circle_outline(center_x, center_y, radius, segments=36, line_width=2):
    """Draw a circle outline"""
    glLineWidth(line_width)
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        glVertex2f(x, y)
    glEnd()

def visualize_grid_env_hd(env, episodes=3, fps=3, tile_size=140, model=None):
    
    # Grid shape
    GRID_HEIGHT = getattr(env, 'grid_size', getattr(env, 'grid_height', 5))
    GRID_WIDTH = getattr(env, 'grid_size', getattr(env, 'grid_width', 5))
    WIDTH = tile_size * GRID_WIDTH
    HEIGHT = tile_size * GRID_HEIGHT + 120  # More space for info
    
    TILE_TYPES = getattr(env, 'TILE_TYPES', {})
    if not TILE_TYPES:
        TILE_TYPES = getattr(env, 'tile_types', {})

    # State variables
    step_count = 0
    episode_count = 1
    done = False
    reward = 0
    total_reward = 0
    episode_rewards = []

    obs, _ = env.reset()

    def draw_text(x, y, text, font=GLUT_BITMAP_HELVETICA_18, color=BLACK):
        glColor3fv(color)
        glRasterPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(font, ord(ch))

    def draw_gradient_quad(x1, y1, x2, y2, color1, color2):
        """Draw a gradient quad for better visual appeal"""
        glBegin(GL_QUADS)
        glColor3fv(color1)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glColor3fv(color2)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()

    def draw_grid():
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x = col * tile_size
                y = row * tile_size
                
                # Tile color and type
                tile_color = GREEN
                tile_type = None
                
                if hasattr(env, "goal") and (row, col) == env.goal:
                    tile_color = RED
                    tile_type = "goal"
                elif hasattr(env, "obstacles") and (row, col) in env.obstacles:
                    tile_color = BLUE
                    tile_type = "obstacle"
                elif (row, col) in TILE_TYPES and TILE_TYPES[(row, col)] == "Module":
                    tile_color = LIGHT_GREEN
                    tile_type = "module"
                
                if tile_type == "goal":
                    draw_gradient_quad(x, y, x + tile_size, y + tile_size, 
                                     norm((255, 120, 90)), norm((255, 80, 50)))
                elif tile_type == "obstacle":
                    draw_gradient_quad(x, y, x + tile_size, y + tile_size,
                                     norm((120, 160, 250)), norm((80, 130, 220)))
                elif tile_type == "module":
                    draw_gradient_quad(x, y, x + tile_size, y + tile_size,
                                     norm((220, 255, 220)), norm((180, 245, 180)))
                else:
                    # Regular tile
                    glColor3fv(tile_color)
                    glBegin(GL_QUADS)
                    glVertex2f(x, y)
                    glVertex2f(x + tile_size, y)
                    glVertex2f(x + tile_size, y + tile_size)
                    glVertex2f(x, y + tile_size)
                    glEnd()
                
                glColor3fv(DARK_GREY)
                glLineWidth(3)
                glBegin(GL_LINE_LOOP)
                glVertex2f(x, y)
                glVertex2f(x + tile_size, y)
                glVertex2f(x + tile_size, y + tile_size)
                glVertex2f(x, y + tile_size)
                glEnd()
                
                # Perfect centering
                if (row, col) in TILE_TYPES:
                    label = TILE_TYPES[(row, col)]
                
                    char_width = 8
                    label_width = len(label) * char_width
                    label_x = x + (tile_size - label_width) // 2
                    label_y = y + tile_size // 2 - 8
                    
                    draw_text(label_x + 1, label_y + 1, label, GLUT_BITMAP_HELVETICA_12, DARK_GREY)
                    draw_text(label_x, label_y, label, GLUT_BITMAP_HELVETICA_12, BLACK)

    def draw_agent():
        if hasattr(env, "agent_pos"):
            ax = env.agent_pos[1] * tile_size
            ay = env.agent_pos[0] * tile_size
            
            center_x = ax + tile_size // 2
            center_y = ay + tile_size // 2
            radius = tile_size // 2.8
            
            # Outer glow effect
            glColor4f(1.0, 0.9, 0.0, 0.3)  # Semi-transparent yellow
            draw_circle(center_x, center_y, radius + 8, 40)

            glColor3fv(YELLOW)
            draw_circle(center_x, center_y, radius, 40)
            
            # Inner highlight
            glColor3fv(norm((255, 255, 200)))
            draw_circle(center_x - radius//4, center_y - radius//4, radius//3, 20)
            
            # Border
            glColor3fv(BLACK)
            draw_circle_outline(center_x, center_y, radius, 40, 3)
            
            label = "Creative"
            char_width = 7  # Character width for HELVETICA_12
            label_width = len(label) * char_width
            label_x = center_x - label_width // 2
            label_y = center_y - 6  # Vertical center adjustment
            
            # Text with shadow for better visibility
            draw_text(label_x + 1, label_y + 1, label, GLUT_BITMAP_HELVETICA_12, DARK_GREY)
            draw_text(label_x, label_y, label, GLUT_BITMAP_HELVETICA_12, BLACK)

    def draw_info_bar():
        draw_gradient_quad(0, HEIGHT - 120, WIDTH, HEIGHT, 
                          norm((240, 240, 240)), norm((255, 255, 255)))
        
        # Subtle border
        glColor3fv(DARK_GREY)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        glVertex2f(0, HEIGHT - 120)
        glVertex2f(WIDTH, HEIGHT - 120)
        glEnd()
        
        margin = 20
        line_height = 25
        
        draw_text(margin, HEIGHT - 90, 
                 f"Episode: {episode_count}/{episodes}   Step: {step_count}", 
                 GLUT_BITMAP_HELVETICA_18, BLACK)
        
        # Rewards info
        draw_text(margin, HEIGHT - 65, 
                 f"Current Reward: {reward:.1f}   Total Reward: {total_reward:.1f}", 
                 GLUT_BITMAP_HELVETICA_18, BLACK)
        
        if episode_rewards:
            rewards_text = "Episode Rewards: " + ", ".join([f"{r:.1f}" for r in episode_rewards[-5:]])
            if len(episode_rewards) > 5:
                rewards_text += "..."
            draw_text(margin, HEIGHT - 40, rewards_text, GLUT_BITMAP_HELVETICA_12, DARK_GREY)
        
        # Progress indicator
        if episodes > 1:
            progress = (episode_count - 1 + (step_count / 25)) / episodes
            progress_width = WIDTH - 2 * margin
            progress_x = margin
            progress_y = HEIGHT - 20
            
            # Progress bar background
            glColor3fv(GREY)
            glBegin(GL_QUADS)
            glVertex2f(progress_x, progress_y)
            glVertex2f(progress_x + progress_width, progress_y)
            glVertex2f(progress_x + progress_width, progress_y + 8)
            glVertex2f(progress_x, progress_y + 8)
            glEnd()
            
            # Progress bar fill
            glColor3fv(norm((0, 150, 0)))
            glBegin(GL_QUADS)
            glVertex2f(progress_x, progress_y)
            glVertex2f(progress_x + progress_width * progress, progress_y)
            glVertex2f(progress_x + progress_width * progress, progress_y + 8)
            glVertex2f(progress_x, progress_y + 8)
            glEnd()

    def display():
        glClear(GL_COLOR_BUFFER_BIT)
        draw_grid()
        draw_agent()
        draw_info_bar()
        glutSwapBuffers()

    def update(_):
        nonlocal obs, step_count, episode_count, done, reward, total_reward, episode_rewards
        
        if done:
            episode_rewards.append(total_reward)
            print(f"Episode {episode_count} completed! Total reward: {total_reward:.1f}")
            
            if episode_count < episodes:
                episode_count += 1
                obs, _ = env.reset()
                step_count = 0
                done = False
                reward = 0
                total_reward = 0
                glutPostRedisplay()
                glutTimerFunc(2000, update, 0)
            else:
                print("\n All episodes completed!")
                print(f"Episode rewards: {[f'{r:.1f}' for r in episode_rewards]}")
                print(f"Average reward: {np.mean(episode_rewards):.1f}")
                print(f"Best episode: {max(episode_rewards):.1f}")
                
                glutPostRedisplay()
                def safe_exit(_):
                    try:
                        glutLeaveMainLoop()
                    except:
                        sys.exit(0)
                glutTimerFunc(5000, safe_exit, 0)
        else:
            if model is not None:
                action, _ = model.predict(np.array(obs).reshape(1, -1), deterministic=True)
                if hasattr(action, 'item'):
                    action = action.item()
            else:
                action = env.action_space.sample()
            
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
            total_reward += reward
            
            glutPostRedisplay()
            glutTimerFunc(int(1000 / fps), update, 0)

    # Initialize OpenGL with maximum quality settings
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_MULTISAMPLE)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"African Creative Learning Journey - HD")
    
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    #glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST)
    
    # 2D rendering
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, HEIGHT, 0)
    glMatrixMode(GL_MODELVIEW)
    glClearColor(0.98, 0.98, 0.98, 1.0)  # Almost white background
    
    glutDisplayFunc(display)
    glutTimerFunc(int(1000 / fps), update, 0)
    
    print(f" Starting HD visualization of {episodes} episodes...")
    print("Enhanced graphics with perfect centering and anti-aliasing!")
    glutMainLoop()

# HD
def visualize_grid_env(env, episodes=3, fps=3, tile_size=140, model=None):
    return visualize_grid_env_hd(env, episodes, fps, tile_size, model)
