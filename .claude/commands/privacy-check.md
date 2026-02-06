# Privacy Compliance Check

You are auditing the privacy compliance of the current session.

## Instructions

1. Review the audit log for this session
2. Check that:
   - All data access went through MCP data server tools
   - No raw patient data appears in the conversation
   - Cell suppression was applied to all small counts
   - No PHI was detected in any outputs
3. Report findings:
   - Number of data server tool calls
   - Number of privacy suppressions applied
   - Any potential privacy concerns
   - Compliance status: PASS/WARN/FAIL

## Audit Checks
- [ ] All data access via MCP tools (not direct file access)
- [ ] No individual patient records in conversation
- [ ] All counts >= minimum cell size (10)
- [ ] No PHI (names, MRNs, DOBs) in any output
- [ ] Audit trail is complete

## Privacy Reminder
This skill itself should never access raw data - only the audit log.
