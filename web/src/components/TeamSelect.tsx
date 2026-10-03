import { SelectField } from './Controls'

export function TeamSelect({ teams, value, onChange, label = 'Team' }: {
  teams: { team_id: number; team_name: string }[]; value?: number; onChange: (id: number | null) => void; label?: string;
}) {
  return <SelectField label={label} value={value ?? ''} onChange={e => onChange(e.target.value === '' ? null : Number(e.target.value))}>
    <option value="">Choose a team…</option>{teams.map(team => <option key={team.team_id} value={team.team_id}>{team.team_name}</option>)}
  </SelectField>
}
