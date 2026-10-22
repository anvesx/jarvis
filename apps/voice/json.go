package main

import "encoding/json"

// json_ decodes raw into v and reports whether it worked.
func json_(raw json.RawMessage, v any) bool { return json.Unmarshal(raw, v) == nil }
