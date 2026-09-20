const fs = require("fs");
const crypto = require("crypto");
const { chromium } = require("playwright");

const req = JSON.parse(fs.readFileSync("automation/post.json", "utf8"));
const login = process.env.FB_LOGIN;
const password = process.env.FB_PASSWORD;

if (!login || !password) {
  console.error("Missing FB_LOGIN or FB_PASSWORD.");
  process.exit(2);
}
if (!req || typeof req.text !== "string" || !req.text.trim()) {
  console.error("automation/post.json must contain a non-empty text field.");
  process.exit(2);
}

const target = req.target || "https://www.facebook.com/";
const requestId = req.request_id || "unspecified";
const text = req.text;
const textHash = crypto.createHash("sha256").update(text).digest("hex");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    locale: "en-US",
    viewport: { width: 1365, height: 900 },
  });
  const page = await context.newPage();

  try {
    await page.goto("https://www.facebook.com/", {
      waitUntil: "domcontentloaded",
      timeout: 60000,
    });

    const email = page.locator('input[name="email"]').first();
    const pass = page.locator('input[name="pass"]').first();

    if (await email.isVisible().catch(() => false)) {
      await email.fill(login);
      await pass.fill(password);

      const loginButton = page.locator('button[name="login"]').first();
      if (await loginButton.isVisible().catch(() => false)) {
        await loginButton.click();
      } else {
        await page.getByRole("button", { name: /log in/i }).first().click();
      }

      await page.waitForLoadState("domcontentloaded", { timeout: 60000 }).catch(() => {});
      await page.waitForTimeout(4000);
    }

    const url = page.url();
    const bodyText = await page.locator("body").innerText().catch(() => "");

    if (
      /checkpoint|two_factor|recover|login/identify/i.test(url) ||
      /two-factor|security check|enter code|confirm your identity|approval needed/i.test(bodyText)
    ) {
      throw new Error("FACEBOOK_AUTH_CHALLENGE");
    }

    if (await page.locator('input[name="email"]').first().isVisible().catch(() => false)) {
      throw new Error("FACEBOOK_LOGIN_NOT_ESTABLISHED");
    }

    await page.goto(target.replace("m.facebook.com", "www.facebook.com"), {
      waitUntil: "domcontentloaded",
      timeout: 60000,
    });
    await page.waitForTimeout(3500);

    const composerCandidates = [
      page.getByText(/What's on your mind/i).first(),
      page.getByRole("button", { name: /What's on your mind/i }).first(),
    ];

    let opened = false;
    for (const candidate of composerCandidates) {
      if (await candidate.isVisible().catch(() => false)) {
        await candidate.click();
        opened = true;
        break;
      }
    }
    if (!opened) {
      throw new Error("FACEBOOK_COMPOSER_NOT_FOUND");
    }

    const dialog = page.getByRole("dialog").last();
    await dialog.waitFor({ state: "visible", timeout: 15000 });

    let textbox = dialog.locator('[role="textbox"][contenteditable="true"]').first();
    if (!(await textbox.isVisible().catch(() => false))) {
      textbox = dialog.locator('[contenteditable="true"]').first();
    }
    await textbox.waitFor({ state: "visible", timeout: 15000 });
    await textbox.fill(text);

    const postButton = dialog.getByRole("button", { name: /^Post$/i }).last();
    await postButton.waitFor({ state: "visible", timeout: 15000 });
    await postButton.click();

    await dialog.waitFor({ state: "hidden", timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(5000);

    const article = page.locator('div[role="article"]').filter({ hasText: text }).first();
    const verified = await article.isVisible().catch(() => false);

    if (!verified) {
      throw new Error("FACEBOOK_POST_NOT_VERIFIED");
    }

    console.log(
      `FACEBOOK_POST_VERIFIED request_id=${requestId} text_sha256=${textHash} final_url=${page.url()}`
    );
  } catch (err) {
    console.error(`FACEBOOK_POST_FAILED request_id=${requestId} code=${err.message}`);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
