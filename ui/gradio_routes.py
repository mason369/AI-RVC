"""Find Gradio handlers in both eager and lazy FastAPI router trees.

FastAPI 0.137 preserves included routers instead of cloning their leaf routes:
https://github.com/fastapi/fastapi/releases/tag/0.137.0
Only traverse the tree; keep the original handlers, dependencies and prefixes.
"""


def iter_gradio_routes(routes):
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            yield from iter_gradio_routes(included.routes)
        else:
            yield route
