#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把刚生成的日报 (public/index.html) 通过 Gmail SMTP 发到邮箱
==========================================================

由 CI 在日报发布后调用。凭据走环境变量（GitHub Secrets），不入库：

- ``MAIL_USERNAME``  发件 Gmail 地址
- ``MAIL_PASSWORD``  该账号的 Google「应用专用密码」(16 位)，不是登录密码
- ``MAIL_TO``        收件地址，可选，缺省发给 ``MAIL_USERNAME`` 自己
- ``REPORT_URL``     网页版链接，可选，放在邮件顶部

未配置凭据时只告警并正常退出，不让日报流水线变红；配置了但发送失败则非零退出。

运行
    python code/send_email.py            # 发送
    python code/send_email.py --dry-run  # 只生成 public/email_preview.eml，不发送
"""
import os
import re
import smtplib
import sys
from email.message import EmailMessage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(ROOT, "public", "index.html")
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465


def inline_css_vars(html: str) -> str:
    """Gmail 不支持 CSS 变量，把 var(--x) 替换成 :root 里定义的字面值。"""
    root = re.search(r":root\s*\{([^}]*)\}", html)
    if not root:
        return html
    values = dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root.group(1)))
    return re.sub(r"var\((--[\w-]+)\)", lambda m: values.get(m.group(1), m.group(0)).strip(), html)


def build_message(html: str, sender: str, to: str, url: str = "") -> EmailMessage:
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    msg = EmailMessage()
    msg["Subject"] = title.group(1).strip() if title else "每日投资监控雷达"
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(f"日报已生成，请用支持 HTML 的邮件客户端查看。\n{url}".strip())

    body = inline_css_vars(html)
    if url:
        link = (f'<p style="max-width:850px;margin:0 auto 8px;font-size:12px;">'
                f'<a href="{url}">在浏览器中打开网页版</a></p>')
        body = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + link, body, count=1)
    msg.add_alternative(body, subtype="html")
    return msg


def main() -> int:
    dry_run = "--dry-run" in sys.argv[1:]
    sender = os.environ.get("MAIL_USERNAME", "").strip()
    password = os.environ.get("MAIL_PASSWORD", "").replace(" ", "")
    to = os.environ.get("MAIL_TO", "").strip() or sender
    url = os.environ.get("REPORT_URL", "").strip()

    with open(REPORT_PATH, encoding="utf-8") as f:
        html = f.read()

    if dry_run:
        msg = build_message(html, sender or "sender@example.com", to or "to@example.com", url)
        out = os.path.join(ROOT, "public", "email_preview.eml")
        with open(out, "wb") as f:
            f.write(msg.as_bytes())
        print(f"✅ 预览已写入 {out}（未发送）")
        return 0

    if not sender or not password:
        print("::warning::未配置 MAIL_USERNAME / MAIL_PASSWORD，跳过邮件发送")
        return 0

    msg = build_message(html, sender, to, url)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    print(f"✅ 日报邮件已发送：{msg['Subject']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
