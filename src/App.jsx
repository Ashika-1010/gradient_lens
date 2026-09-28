export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <div className="max-w-2xl w-full bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-2xl">
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-4 w-4 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-xs uppercase tracking-widest text-emerald-400 font-semibold font-mono">
            Milestone 0: Project Setup
          </span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">
          Gradient Lens
        </h1>
        <p className="text-slate-400 text-lg mb-6">
          See exactly why a weight changed.
        </p>
        
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-4 font-mono text-sm space-y-2 text-slate-300">
          <div className="flex justify-between border-b border-slate-800 pb-2">
            <span className="text-slate-500">Framework:</span>
            <span className="text-slate-200">React + Vite (JS)</span>
          </div>
          <div className="flex justify-between border-b border-slate-800 pb-2">
            <span className="text-slate-500">Styling:</span>
            <span className="text-slate-200">Tailwind CSS v4</span>
          </div>
          <div className="flex justify-between border-b border-slate-800 pb-2">
            <span className="text-slate-500">Test Runner:</span>
            <span className="text-slate-200">Vitest</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Engine Sandbox:</span>
            <span className="text-emerald-400">Pure JavaScript (src/engine/)</span>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-800 text-xs text-slate-500 flex justify-between items-center">
          <span>Target Architecture: 2 → 2 → 1 (XOR)</span>
          <span>Status: Ready for Milestone 1</span>
        </div>
      </div>
    </div>
  )
}
