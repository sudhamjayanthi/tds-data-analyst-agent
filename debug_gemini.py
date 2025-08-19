import os
import asyncio
import logging
import google.generativeai as genai
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-pro")

    async def log_response(tag: str, response):
        logger.info(f"=== {tag} ===")
        logger.info(f"Raw: {response}")
        try:
            logger.info(
                f"candidates: {len(response.candidates) if response.candidates else 0}"
            )
            for i, cand in enumerate(response.candidates or []):
                logger.info(
                    f"candidate[{i}].role: {getattr(cand.content, 'role', 'unknown')}"
                )
                for j, part in enumerate(cand.content.parts or []):
                    if hasattr(part, "text") and part.text:
                        logger.info(f"part[{j}].text[:200]: {part.text[:200]}")
                    elif hasattr(part, "function_call") and part.function_call:
                        logger.info(
                            f"part[{j}].function_call.name: {getattr(part.function_call, 'name', None)}"
                        )
                    else:
                        logger.info(f"part[{j}].type: {type(part)}")
        except Exception as e:
            logger.exception(f"Failed to introspect response: {e}")

    # 1) Pure text prompt
    r1 = await model.generate_content_async("Say hello in one short sentence.")
    await log_response("TEXT", r1)

    # 2) Ask for python code (no tools)
    r2 = await model.generate_content_async(
        "Write a short python snippet that prints 42"
    )
    await log_response("CODE_NO_TOOLS", r2)

    # 3) With tools: force a function call-like behavior
    async def dummy(code: str) -> str:
        return "{}"

    model_tools = genai.GenerativeModel("gemini-2.5-pro", tools=[dummy])
    r3 = await model_tools.generate_content_async(
        "Call the dummy tool with an argument code='print(1)'"
    )
    await log_response("WITH_TOOLS", r3)


if __name__ == "__main__":
    asyncio.run(main())
