package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

func TestSessionFor(t *testing.T) {
	if got := sessionFor("orb-20261021-191500-a1b2c3"); got != "20261021-191500-a1b2c3" {
		t.Fatalf("got %q", got)
	}
	a, b := sessionFor("Some Room!"), sessionFor("Some Room!")
	if a != b || !sessionID.MatchString(a) {
		t.Fatalf("a foreign room must map to one valid, stable session: %q %q", a, b)
	}
	if sessionFor("orb-BAD ID") == "BAD ID" {
		t.Fatal("an invalid ID after the prefix must not pass")
	}
}

func TestLoadDotenvKeepsSetVariables(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	os.WriteFile(path, []byte("# c\nORB_T_A=file\nexport ORB_T_B=\"two\"\nORB_T_EMPTY=\n"), 0o600)
	t.Setenv("ORB_T_A", "process")
	os.Unsetenv("ORB_T_B")
	loadDotenv(path)
	if os.Getenv("ORB_T_A") != "process" || os.Getenv("ORB_T_B") != "two" {
		t.Fatalf("A=%q B=%q", os.Getenv("ORB_T_A"), os.Getenv("ORB_T_B"))
	}
	os.Unsetenv("ORB_T_B")
}

func TestWriteWAVHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.wav")
	f := audio.NewFrame([]int16{1, -1, 300, -300}, 16000, 1)
	if err := writeWAV(path, []audio.Frame{f, f}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if len(b) != 44+16 || string(b[0:4]) != "RIFF" || string(b[8:16]) != "WAVEfmt " || string(b[36:40]) != "data" {
		t.Fatalf("bad WAV: %d bytes, %q", len(b), b[:16])
	}
	if rate := uint32(b[24]) | uint32(b[25])<<8 | uint32(b[26])<<16 | uint32(b[27])<<24; rate != 16000 {
		t.Fatalf("rate %d", rate)
	}
}
