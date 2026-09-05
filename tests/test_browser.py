"""Opt-in browser regression: catches frontend failures Python rendering misses.

Start Solara and set MAS_BROWSER_URL to run. MAS_BROWSER_EXECUTABLE optionally
selects an installed Chrome; otherwise Playwright uses its managed Chromium.
"""

import os
import re
import time
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.skipif(not os.environ.get("MAS_BROWSER_URL"), reason="Requires a running Solara server")
def test_fish_visible_and_controls_update_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(
            executable_path=os.environ.get("MAS_BROWSER_EXECUTABLE"), headless=True
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(os.environ["MAS_BROWSER_URL"], wait_until="domcontentloaded")
            images = page.locator("img.widget-image")
            images.first.wait_for(state="visible", timeout=45000)
            images.nth(1).wait_for(state="visible", timeout=10000)
            page.wait_for_function(
                "Array.from(document.querySelectorAll('img.widget-image')).every(i => i.complete && i.naturalWidth > 0)"
            )
            tank = images.first

            def wait_for_tank(predicate):
                # Poll the fetched SVG, rather than a blob URL that can change
                # before the simulation update has reached the browser.
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    rendered = tank.evaluate("async image => (await fetch(image.src)).text()")
                    if predicate(rendered):
                        return rendered
                    page.wait_for_timeout(100)
                pytest.fail("The tank did not render the expected simulation update")

            def simulation_time(rendered):
                return float(re.search(r"fish · ([\d.]+) s", rendered).group(1))

            svg = tank.evaluate("async image => (await fetch(image.src)).text()")
            assert "Preferred light: 25%" in svg
            assert "Light intensity (%)" in svg
            root = ET.fromstring(svg)
            quiver = next(node for node in root.iter() if node.attrib.get("id", "").startswith("Quiver_"))
            assert len(quiver) == 60, "The tank must contain all 60 fish arrows"
            assert tank.bounding_box()["width"] > 200

            # Perturbations must be visible immediately, including while paused.
            page.get_by_role("button", name="Predator attack", exact=True).click()
            wait_for_tank(lambda rendered: "PREDATOR" in rendered)
            page.get_by_role("button", name="Storm", exact=True).click()
            wait_for_tank(lambda rendered: "STORM" in rendered)
            page.get_by_role("button", name="Clear disturbances", exact=True).click()
            wait_for_tank(lambda rendered: "PREDATOR" not in rendered and "STORM" not in rendered)

            page.get_by_role("button", name="Step", exact=True).click()
            after_step = wait_for_tank(lambda rendered: simulation_time(rendered) > 0)
            step_time = simulation_time(after_step)
            updated_root = ET.fromstring(after_step)
            updated_fish = next(node for node in updated_root.iter() if node.attrib.get("id", "").startswith("Quiver_"))
            assert [node.attrib["d"] for node in updated_fish] != [node.attrib["d"] for node in quiver]

            page.get_by_role("button", name="▶", exact=True).click()
            pause = page.get_by_role("button", name="❚❚", exact=True)
            pause.wait_for()
            wait_for_tank(lambda rendered: simulation_time(rendered) > step_time)
            pause.click()
            # Start with light enabled, then check that switching it off and on
            # updates both the environment and the preferred-light overlay.
            page.get_by_text("Enable light gradient", exact=True).click()
            playwright.expect(page.get_by_label("Enable light gradient")).not_to_be_checked()
            page.get_by_role("button", name="Reset", exact=True).click()
            wait_for_tank(lambda rendered: "BRIGHT" not in rendered and simulation_time(rendered) == 0)
            # Vuetify places its clickable ripple over the native input.
            page.get_by_text("Enable light gradient", exact=True).click()
            playwright.expect(page.get_by_label("Enable light gradient")).to_be_checked()
            page.get_by_role("button", name="Reset", exact=True).click()
            wait_for_tank(lambda rendered: "BRIGHT" in rendered and simulation_time(rendered) == 0)
            assert not errors, "Browser errors: " + "; ".join(errors)
        finally:
            browser.close()
