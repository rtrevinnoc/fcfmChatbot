import asyncio
import httpx
import re

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://www.fcfm.uanl.mx/avisos/")
        text = resp.text
        # Find all URLs matching https://www.fcfm.uanl.mx/avisos/[a-z0-9-]+/
        matches = re.findall(r'href=["\'](https://www\.fcfm\.uanl\.mx/avisos/[^"\']+)["\']', text)
        print("Found URLs:")
        for m in sorted(set(matches)):
            print(m)

asyncio.run(test())
