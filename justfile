set shell := ["bash", "-euo", "pipefail", "-c"]

# +----------------------------------------------------------------------------+
# | Sandbox — run Claude Code inside a Docker Sandboxes microVM                |
# +----------------------------------------------------------------------------+

# Every recipe below is a literal `sbx` command line with the sandbox name
# spelled out, so a line can be pasted into a shell or a script unchanged and
# `just` stays a convenience rather than a dependency. What the sandbox installs
# is declared in .sbx/kit/spec.yaml, not here.

# A kit only takes effect at creation, so changing it means recreating the
# sandbox. Nothing is lost: the Claude Code configuration and session history the
# kit puts under .sbx/ live on the host and outlive the sandbox.
#
# The denied ranges are what keeps the sandbox off the LAN: the three RFC 1918
# private blocks, RFC 3927 link-local (where cloud metadata endpoints such as
# 169.254.169.254 sit), and RFC 6598 shared address space, which Tailscale
# numbers tailnets from, then the IPv6 unique-local and link-local blocks. They
# are passed here rather than declared in the kit because a kit declares CIDR
# rules without applying them.
#
# A LAN numbered from globally routable IPv6 would still be reachable. Closing
# that means denying ::/0, which also gives up IPv6 egress — fine for a sandbox
# that only needs package registries, but only if the fallback to IPv4 holds.
# Build and start the sandbox, replacing any existing one
sbx-up:
    sbx rm --force plaquette || true
    sbx create --name plaquette --kit ./.sbx/kit \
        --deny-network 10.0.0.0/8 \
        --deny-network 172.16.0.0/12 \
        --deny-network 192.168.0.0/16 \
        --deny-network 169.254.0.0/16 \
        --deny-network 100.64.0.0/10 \
        --deny-network fc00::/7 \
        --deny-network fe80::/10 \
        claude .

# Authenticate Claude Code and store the token in the sbx keychain. Needed once
# per login, not per sandbox.
# Sign in to Claude Code inside the sandbox
sbx-login:
    sbx run --name plaquette -- auth login

# Attach Claude Code to the running sandbox
sbx-agent:
    sbx run --name plaquette

# Open a login shell in the running sandbox
sbx-shell:
    sbx exec -it plaquette bash -l
