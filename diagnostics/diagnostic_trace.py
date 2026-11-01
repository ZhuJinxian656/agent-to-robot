"""Private Python-cell/capability boundaries; never modify a policy response.

Installed only by the reproduction's transfer entry point. Read completed call
indices incrementally when the existing private tool audit writes cell boundaries.
No hidden outcome, observation, prompt, or receipt is added to the policy context.
"""
import json
import time


def install():
    from captain_harness.react.tools import ToolExecutionAudit
    if getattr(ToolExecutionAudit, '_repro_diagnostics_installed', False):
        return
    original = ToolExecutionAudit.write

    def write(self, event):
        with self._lock:
            row = dict(event)
            trace = dict(version='private-cell-capability-boundary/v1', timestamp_wall_ns=time.time_ns())
            if event.get('type') in ('tool_start', 'tool_end') and event.get('tool_name') == 'python':
                path = self.path.parent.parent/'calls.jsonl'
                offset = getattr(self, '_repro_call_offset', 0)
                index = getattr(self, '_repro_last_call_index', 0)
                try:
                    if path.exists():
                        with path.open('rb') as stream:
                            stream.seek(offset)
                            while True:
                                line = stream.readline()
                                if not line or not line.endswith(b'\n'):
                                    break
                                index = json.loads(line)['call_index']
                                offset = stream.tell()
                    self._repro_call_offset, self._repro_last_call_index = offset, index
                    trace.update(call_index_max=index, active_python_tools=len(self._active_tools))
                except Exception as exc:
                    # Diagnostic failure must not change model execution.
                    trace['boundary_error'] = f'{type(exc).__name__}: {exc}'
            row['host_trace'] = trace
            return original(self, row)

    ToolExecutionAudit.write = write
    ToolExecutionAudit._repro_diagnostics_installed = True
