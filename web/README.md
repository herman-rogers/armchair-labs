# Patron Saints — frontend

React + TypeScript board viewer, served by Vite. Reads the FastAPI read API in
`src/patron/api/`; it computes nothing itself.

```bash
npm install
npm run dev      # :5173, proxies /api to the backend on :8000
npm run build    # type-check and bundle to dist/
```

The backend must be running (`just api` from the repo root) or the page shows a
notice explaining what to start.

## Layout

| Path | What |
|---|---|
| `src/api/types.ts` | Response shapes, mirroring `patron.pipeline.BOARD_EXPORT_COLUMNS` |
| `src/api/client.ts` | Fetch wrappers |
| `src/components/BoardTable.tsx` | The sortable board. Column definitions carry the tooltip text explaining each metric |
| `src/components/Flags.tsx` | BUY / TD-luck / age / Ngms chips |
| `src/index.css` | Everything visual — dark, dense, tabular figures |

## Note on Vite

Pinned to Vite 7 rather than 8. Vite 8 bundles with rolldown, which needs Node 22+;
this machine runs Node 21.7.3. Move to Vite 8 once Node is upgraded.
