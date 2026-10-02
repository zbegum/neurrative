"""The ride GIF for the blog post: the mood page in ride mode, stepped through the book.

Needs playwright (pip install playwright) and Google Chrome, and the page built by
surface/mood/mood_viewer.py.

python docs/blog/ride_gif.py
"""

import io
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAGE = "file://" + os.path.join(ROOT, "surface", "mood", "output", "mood_viewer.html")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BOOK, N, STEP, SCALE, FRAME_S = "alice_wonderland", 789, 3, 0.5, 0.1


def main():
  with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, args=[
      "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars"])
    page = browser.new_page(viewport={"width": 1400, "height": 820})
    page.goto(PAGE)
    page.wait_for_timeout(4000)
    page.click("#ride")
    page.wait_for_timeout(1500)
    box = page.locator("#stage").bounding_box()
    frames = []
    # Step with the arrow key (one paragraph per press): the page moves the rider
    # along the route without redrawing the scene, so consecutive frames are smooth.
    page.locator("#stage").click()
    for i in range(0, N, STEP):
      img = Image.open(io.BytesIO(page.screenshot(clip=box))).convert("RGB")
      frames.append(np.array(img.resize((int(img.width * SCALE), int(img.height * SCALE)),
                                        Image.LANCZOS)))
      for _ in range(STEP):
        page.keyboard.press("ArrowRight")
      page.wait_for_timeout(60)
    browser.close()
  out = os.path.join(HERE, "9_ride.gif")
  imageio.mimsave(out, frames, duration=FRAME_S, loop=0)
  print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
  main()
