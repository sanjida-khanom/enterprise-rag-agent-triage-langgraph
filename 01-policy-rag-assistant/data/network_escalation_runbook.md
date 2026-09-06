# Network Fault Escalation Runbook
Document ID: TECH-RB-027 | Effective: 15 February 2026 | Owner: Network Operations

## 1. Severity Definitions
Severity 1 (S1): Complete site outage affecting more than 5,000 subscribers, or any core network element failure. Response target 15 minutes, resolution target 4 hours.

Severity 2 (S2): Degraded service affecting 1,000 to 5,000 subscribers, or single sector down. Response target 30 minutes, resolution target 8 hours.

Severity 3 (S3): Localised degradation affecting fewer than 1,000 subscribers. Response target 2 hours, resolution target 48 hours.

## 2. Escalation Path
S1 faults escalate immediately to the NOC Shift Lead and the Head of Network Operations. If unresolved after 2 hours, escalate to the Chief Technology Officer.

S2 faults escalate to the NOC Shift Lead. If unresolved after 6 hours, escalate to the Head of Network Operations.

## 3. Common Alarm Codes
Alarm code NE-4021 indicates transmission link failure on the backhaul. First action is to verify the microwave link status and check for power failure at the far end site.

Alarm code NE-3310 indicates a baseband unit temperature threshold breach. Verify air conditioning at the site before considering hardware replacement.

Alarm code NE-5502 indicates VSWR threshold exceeded, typically caused by a damaged feeder cable or antenna connector after adverse weather.

## 4. Power Failure Protocol
Sites running on battery backup must be reported to the Field Operations team when remaining backup falls below 90 minutes. Generator deployment is authorised by the Regional Field Manager.

## 5. Post-Incident Review
All S1 incidents require a root cause analysis document submitted within 5 working days of resolution.
