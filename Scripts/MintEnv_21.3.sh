#!/bin/bash

<<COMMENT
1. Run this script using the following command:
wget -O - https://raw.githubusercontent.com/sarisabban/Notes/master/Scripts/MintEnv_21.3.sh | bash
2. Setup FireFox
COMMENT

# MINT 21.3 ENVIRONMENT SETUP
#============================

# Install Applications:
#----------------------
sudo apt update
sudo apt full-upgrade -y
sudo apt install weechat vim tmux htop git python3-venv -y

# Setup Terminal Environment:
#----------------------------
n="$(awk '/PS1=/ { $0 = "PS1=\x27\\[\\033[0;33m\\][\\W]\\$ \\[\\033[0m\\]\x27" } 1' .bashrc)"; echo "$n" > ~/.bashrc
printf ':set shiftwidth=4 noexpandtab\n:set softtabstop=4 noexpandtab\n:set list\n:set listchars=tab:»·,trail:·,extends:>,precedes:<,nbsp:+\n:syntax enable\n:set nowrap\n:set number\n:set ts=4 sw=4\n:set cc=81' >> ~/.vimrc
printf "set -g status-bg black\nset -g status-fg red\nset -g pane-border-style fg=magenta\nset -g pane-active-border-style fg=magenta\nset -g status-justify 'centre'\nset -g status-left-length '100'\nset -g status 'on'\nset -g status-right-length '100'\nsetw -g window-status-separator ''\nset -g status-right '%%H:%%M'\nsetw -g window-status-current-format '#(whoami)'\nset -g status off" > ~/.tmux.conf

# Turn Off Bluetooth:
#--------------------
rfkill block bluetooth

# Setup Region & Langauge Formats:
#---------------------------------
sudo update-locale LC_NUMERIC=en_GB.UTF-8
sudo update-locale LC_MONETARY=en_GB.UTF-8
sudo update-locale LC_MEASUREMENT=en_GB.UTF-8
sudo update-locale LC_ADDRESS=en_GB.UTF-8
sudo update-locale LC_IDENTIFICATION=en_GB.UTF-8
sudo update-locale LC_TIME=en_GB.UTF-8
sudo update-locale LC_TELEPHONE=en_GB.UTF-8
sudo update-locale LC_NAME=en_GB.UTF-8
sudo update-locale LANG=en_GB.UTF-8
sudo update-locale LC_PAPER=en_GB.UTF-8

# Setup htop Environment:
#------------------------
mkdir ~/.config/htop/
touch ~/.config/htop/htoprc
echo 'fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=-1
tree_sort_key=0
tree_sort_direction=1
hide_kernel_threads=1
hide_userland_threads=0
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
highlight_changes=0
highlight_changes_delay_secs=5
find_comm_in_cmdline=1
strip_exe_from_cmdline=1
show_merged_command=0
tree_view=0
tree_view_always_by_pid=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_one=0
show_cpu_usage=1
show_cpu_frequency=0
show_cpu_temperature=1
degree_fahrenheit=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
enable_mouse=0
delay=15
left_meters=AllCPUs Memory Swap Battery
left_meter_modes=1 1 1 1
right_meters=NetworkIO NetworkIO Tasks Uptime
right_meter_modes=3 2 2 2
hide_function_bar=0'>>~/.config/htop/htoprc

# Setup Remaining Distro Environment:
#------------------------------------
echo "[com/linuxmint/updates]
refresh-last-run=1721554121
window-height=570
window-pane-position=344
window-width=790

[org/blueman/general]
window-properties=[500, 350, 50, 82]

[org/cinnamon]
alttab-switcher-delay=100
desklet-snap-interval=25
enabled-applets=['panel1:right:3:keyboard@cinnamon.org:8', 'panel1:right:6:calendar@cinnamon.org:15', 'panel1:left:0:menu@cinnamon.org:22', 'panel1:right:4:network@cinnamon.org:23', 'panel1:right:5:power@cinnamon.org:24', 'panel1:left:1:separator@cinnamon.org:25', 'panel1:left:2:grouped-window-list@cinnamon.org:29']
enabled-desklets=@as []
hotcorner-layout=['expo:false:0', 'scale:false:0', 'scale:false:0', 'desktop:false:0']
next-applet-id=30
panel-edit-mode=false
panel-zone-symbolic-icon-sizes='[{"panelId": 1, "left": 28, "center": 28, "right": 16}]'
panels-enabled=['1:0:top']
panels-height=['1:20']
window-effect-speed=1

[org/cinnamon/cinnamon-session]
quit-time-delay=60

[org/cinnamon/desktop/background]
picture-options='zoom'
picture-uri='file:///usr/share/backgrounds/linuxmint-vanessa/fakurian_purple.jpg'

[org/cinnamon/desktop/background/slideshow]
delay=15
image-source='xml:///usr/share/cinnamon-background-properties/linuxmint-vanessa.xml'

[org/cinnamon/desktop/interface]
clock-show-date=false
clock-show-seconds=false
cursor-blink-time=1200
cursor-size=24
cursor-theme='Bibata-Modern-Classic'
first-day-of-week=0
gtk-theme='Mint-Y-Dark-Grey'
icon-theme='Mint-Y-Grey'
keyboard-layout-prefer-variant-names=true
keyboard-layout-show-flags=false
keyboard-layout-use-upper=true
text-scaling-factor=1.0
toolkit-accessibility=false

[org/cinnamon/desktop/keybindings]
custom-list=['custom1', 'custom0', '__dummy__', 'custom2']

[org/cinnamon/desktop/keybindings/custom-keybindings/custom0]
binding=['<Primary><Alt>w']
command='firefox'
name='FireFox'

[org/cinnamon/desktop/keybindings/custom-keybindings/custom1]
binding=['<Primary><Alt>h']
command==
name='Home'

[org/cinnamon/desktop/keybindings/custom-keybindings/custom2]
binding=['<Primary><Alt>g']
command='xed'
name='Text'

[org/cinnamon/desktop/notifications]
display-notifications=false
notification-duration=4

[org/cinnamon/desktop/peripherals/keyboard]
delay=uint32 500
repeat-interval=uint32 30

[org/cinnamon/desktop/peripherals/mouse]
double-click=400
drag-threshold=8
speed=0.0

[org/cinnamon/desktop/peripherals/touchpad]
speed=0.0
tap-to-click=false

[org/cinnamon/desktop/privacy]
remember-recent-files=false

[org/cinnamon/desktop/screensaver]
lock-enabled=false

[org/cinnamon/desktop/sound]
event-sounds=false

[org/cinnamon/desktop/wm/preferences]
min-window-opacity=30
num-workspaces=3
workspace-names=@as []

[org/cinnamon/gestures]
pinch-percent-threshold=uint32 40
swipe-down-2='PUSH_TILE_DOWN::end'
swipe-down-3='TOGGLE_OVERVIEW::end'
swipe-down-4='VOLUME_DOWN::end'
swipe-left-2='PUSH_TILE_LEFT::end'
swipe-left-3='WORKSPACE_NEXT::end'
swipe-left-4='WINDOW_WORKSPACE_PREVIOUS::end'
swipe-percent-threshold=uint32 60
swipe-right-2='PUSH_TILE_RIGHT::end'
swipe-right-3='WORKSPACE_PREVIOUS::end'
swipe-right-4='WINDOW_WORKSPACE_NEXT::end'
swipe-up-2='PUSH_TILE_UP::end'
swipe-up-3='TOGGLE_EXPO::end'
swipe-up-4='VOLUME_UP::end'
tap-3='MEDIA_PLAY_PAUSE::end'

[org/cinnamon/launcher]
check-frequency=300
memory-limit=2048

[org/cinnamon/muffin]
draggable-border-width=10

[org/cinnamon/settings-daemon/peripherals/keyboard]
numlock-state='off'

[org/cinnamon/settings-daemon/plugins/power]
button-power='shutdown'
critical-battery-action='shutdown'
idle-dim-battery=false
lid-close-ac-action='nothing'
lid-close-battery-action='nothing'
lid-close-suspend-with-external-monitor=false
lock-on-suspend=false
sleep-display-ac=0
sleep-display-battery=0
sleep-inactive-ac-timeout=0
sleep-inactive-battery-timeout=0

[org/cinnamon/sounds]
login-enabled=false
logout-enabled=false
notification-enabled=true
plug-enabled=true
switch-enabled=false
tile-enabled=false

[org/cinnamon/theme]
name='Mint-Y-Dark-Grey'

[org/gnome/desktop/a11y]
always-show-text-caret=false

[org/gnome/desktop/a11y/applications]
screen-keyboard-enabled=false
screen-reader-enabled=false

[org/gnome/desktop/a11y/mouse]
dwell-click-enabled=false
dwell-threshold=10
dwell-time=1.2
secondary-click-enabled=false
secondary-click-time=1.2

[org/gnome/desktop/input-sources]
current=uint32 0
show-all-sources=false
sources=@a(ss) []
xkb-options=@as []

[org/gnome/desktop/interface]
can-change-accels=false
clock-format='24h'
clock-show-date=false
clock-show-seconds=false
cursor-blink=true
cursor-blink-time=1200
cursor-blink-timeout=10
cursor-size=24
cursor-theme='Bibata-Modern-Classic'
enable-animations=true
font-name='Ubuntu 10'
gtk-color-palette='black:white:gray50:red:purple:blue:light blue:green:yellow:orange:lavender:brown:goldenrod4:dodger blue:pink:light green:gray10:gray30:gray75:gray90'
gtk-color-scheme=''
gtk-enable-primary-paste=true
gtk-im-module=''
gtk-im-preedit-style='callback'
gtk-im-status-style='callback'
gtk-key-theme='Default'
gtk-theme='Mint-Y-Dark-Grey'
gtk-timeout-initial=200
gtk-timeout-repeat=20
icon-theme='Mint-Y-Grey'
menubar-accel='F10'
menubar-detachable=false
menus-have-tearoff=false
scaling-factor=uint32 0
text-scaling-factor=1.0
toolbar-detachable=false
toolbar-icons-size='large'
toolbar-style='both-horiz'
toolkit-accessibility=false

[org/gnome/desktop/peripherals/mouse]
accel-profile='default'
double-click=400
drag-threshold=8
left-handed=false
middle-click-emulation=false
natural-scroll=false
speed=0.0

[org/gnome/desktop/privacy]
disable-camera=false
disable-microphone=false
disable-sound-output=false
old-files-age=uint32 30
recent-files-max-age=7
remember-recent-files=false
remove-old-temp-files=false
remove-old-trash-files=false

[org/gnome/desktop/sound]
event-sounds=false
input-feedback-sounds=false
theme-name='LinuxMint'

[org/gnome/desktop/wm/preferences]
action-double-click-titlebar='toggle-maximize'
action-middle-click-titlebar='lower'
action-right-click-titlebar='menu'
audible-bell=false
auto-raise=false
auto-raise-delay=500
button-layout=':minimize,maximize,close'
disable-workarounds=false
focus-mode='click'
focus-new-windows='smart'
mouse-button-modifier='<Alt>'
num-workspaces=3
raise-on-click=true
resize-with-right-button=true
theme='Mint-Y'
titlebar-font='Ubuntu Medium 10'
titlebar-uses-system-font=false
visual-bell=false
visual-bell-type='fullscreen-flash'
workspace-names=@as []

[org/gnome/evolution-data-server]
migrated=true
network-monitor-gio-name=''

[org/gnome/libgnomekbd/keyboard]
layouts=['us\tmac', 'ara\tmac']

[org/gnome/settings-daemon/plugins/xsettings]
disabled-gtk-modules=@as []
enabled-gtk-modules=@as []
overrides=@a{sv} {}

[org/gnome/terminal/legacy]
default-show-menubar=false

[org/gnome/terminal/legacy/profiles:/:b1dcc9dd-5262-4d8d-a863-c897e6d979b9]
audible-bell=true
background-color='rgb(0,0,0)'
cursor-shape='underline'
foreground-color='rgb(0,255,0)'
palette=['rgb(0,0,0)', 'rgb(170,0,0)', 'rgb(0,170,0)', 'rgb(170,85,0)', 'rgb(0,0,170)', 'rgb(170,0,170)', 'rgb(0,170,170)', 'rgb(170,170,170)', 'rgb(85,85,85)', 'rgb(255,85,85)', 'rgb(85,255,85)', 'rgb(255,255,85)', 'rgb(85,85,255)', 'rgb(255,85,255)', 'rgb(85,255,255)', 'rgb(255,255,255)']
scrollbar-policy='never'
use-theme-colors=false

[org/gtk/settings/file-chooser]
date-format='regular'
location-mode='path-bar'
show-hidden=false
show-size-column=true
show-type-column=true
sidebar-width=148
sort-column='name'
sort-directories-first=true
sort-order='ascending'
type-format='category'
window-position=(135, 0)
window-size=(1096, 696)

[org/nemo/desktop]
desktop-layout='true::false'
show-orphaned-desktop-icons=true

[org/nemo/list-view]
default-column-order=['name', 'size', 'type', 'date_modified', 'date_created_with_time', 'date_accessed', 'date_created', 'detailed_type', 'group', 'where', 'mime_type', 'date_modified_with_time', 'octal_permissions', 'owner', 'permissions']
default-visible-columns=['name', 'size', 'date_modified']

[org/nemo/preferences]
date-format='iso'
default-folder-viewer='list-view'

[org/nemo/window-state]
geometry='725x515+138+119'
maximized=false
network-expanded=false
side-pane-view='places'
sidebar-bookmark-breakpoint=5
sidebar-width=151
start-with-menu-bar=false
start-with-sidebar=true

[org/x/apps/portal]
color-scheme='prefer-dark'

[org/x/editor/plugins]
active-plugins=['modelines', 'spell', 'time', 'joinlines', 'open-uri-context-menu', 'sort', 'docinfo', 'textsize', 'filebrowser']

[org/x/editor/plugins/filebrowser/on-load]
root='file:///'
tree-view=true
virtual-root='file:///home/slurm/Desktop'

[org/x/editor/preferences/editor]
bracket-matching=false
display-line-numbers=true
draw-whitespace=false
insert-spaces=false
scheme='cobalt'
wrap-mode='none'

[org/x/editor/preferences/ui]
menubar-visible=false
minimap-visible=false
statusbar-visible=false
toolbar-visible=false

[org/x/editor/state/history-entry]
history-search-for=['format']

[org/x/editor/state/window]
bottom-panel-size=140
side-panel-active-page=827629879
side-panel-size=200
size=(683, 716)
state=43908

[org/x/warpinator/preferences]
ask-for-send-permission=true
autostart=false
connect-id='SLURM-B88ED3DD8046057BC50C'
no-overwrite=true
">dconf-settings.ini
printf '%s\n' '/command==/' d i 'command="sh -c '\''setsid xdg-open \"$HOME\" &'\''"' . w q | ed -s dconf-settings.ini
cat dconf-settings.ini | dconf load / # Dump info using: dconf dump / > dconf-settings.ini
rm dconf-settings.ini

# Setup WeeChat Environment:
#---------------------------
weechat &
sleep 3
kill $!
sed -i '/enabled = on/c\enabled = off' ~/.config/weechat/buflist.conf
sed -i '/autoload = "*"/c\autoload = "*,!logger"' ~/.config/weechat/weechat.conf
echo '#
# weechat -- irc.conf
#
# WARNING: It is NOT recommended to edit this file by hand,
# especially if WeeChat is running.
#
# Use commands like /set or /fset to change settings in WeeChat.
#
# For more info, see: https://weechat.org/doc/quickstart
#

[look]
buffer_open_before_autojoin = on
buffer_open_before_join = off
buffer_switch_autojoin = on
buffer_switch_join = on
color_nicks_in_names = off
color_nicks_in_nicklist = off
color_nicks_in_server_messages = on
color_pv_nick_like_channel = on
ctcp_time_format = "%a, %d %b %Y %T %z"
display_account_message = on
display_away = local
display_ctcp_blocked = on
display_ctcp_reply = on
display_ctcp_unknown = on
display_extended_join = on
display_host_join = on
display_host_join_local = on
display_host_quit = on
display_join_message = "329,332,333,366"
display_old_topic = on
display_pv_away_once = on
display_pv_back = on
display_pv_warning_address = off
highlight_channel = "$nick"
highlight_pv = "$nick"
highlight_server = "$nick"
highlight_tags_restrict = "irc_privmsg,irc_notice"
item_channel_modes_hide_args = "k"
item_display_server = buffer_plugin
item_nick_modes = on
item_nick_prefix = on
join_auto_add_chantype = off
msgbuffer_fallback = current
new_channel_position = none
new_pv_position = none
nick_completion_smart = speakers
nick_mode = prefix
nick_mode_empty = off
nicks_hide_password = "nickserv"
notice_as_pv = auto
notice_welcome_redirect = on
notice_welcome_tags = ""
notify_tags_ison = "notify_message"
notify_tags_whois = "notify_message"
part_closes_buffer = off
pv_buffer = independent
pv_tags = "notify_private"
raw_messages = 256
server_buffer = merge_with_core
smart_filter = on
smart_filter_account = on
smart_filter_chghost = on
smart_filter_delay = 5
smart_filter_join = on
smart_filter_join_unmask = 30
smart_filter_mode = "+"
smart_filter_nick = on
smart_filter_quit = on
temporary_servers = off
topic_strip_colors = off
typing_status_nicks = off
typing_status_self = off

[color]
input_nick = lightcyan
item_channel_modes = default
item_lag_counting = default
item_lag_finished = yellow
item_nick_modes = default
item_tls_version_deprecated = yellow
item_tls_version_insecure = red
item_tls_version_ok = green
message_account = cyan
message_chghost = brown
message_join = green
message_kick = red
message_quit = red
mirc_remap = "1,-1:darkgray"
nick_prefixes = "y:lightred;q:lightred;a:lightcyan;o:lightgreen;h:lightmagenta;v:yellow;*:lightblue"
notice = green
reason_kick = default
reason_quit = default
topic_current = default
topic_new = white
topic_old = default

[network]
autoreconnect_delay_growing = 2
autoreconnect_delay_max = 600
ban_mask_default = "*!$ident@$host"
colors_receive = on
colors_send = on
lag_check = 60
lag_max = 1800
lag_min_show = 500
lag_reconnect = 300
lag_refresh_interval = 1
notify_check_ison = 1
notify_check_whois = 5
sasl_fail_unavailable = on
send_unknown_commands = off
whois_double_nick = off

[msgbuffer]

[ctcp]

[ignore]

[server_default]
addresses = ""
anti_flood_prio_high = 2
anti_flood_prio_low = 2
autoconnect = on
autojoin = ""
autojoin_dynamic = off
autoreconnect = on
autoreconnect_delay = 10
autorejoin = off
autorejoin_delay = 30
away_check = 0
away_check_max_nicks = 25
capabilities = "*"
charset_message = message
command = ""
command_delay = 0
connection_timeout = 60
default_chantypes = "#&"
ipv6 = on
local_hostname = ""
msg_kick = ""
msg_part = "WeeChat ${info:version}"
msg_quit = "WeeChat ${info:version}"
nicks = "'$USER','$USER'1,'$USER'2,'$USER'3,'$USER'4"
nicks_alternate = on
notify = ""
password = ""
proxy = ""
realname = ""
sasl_fail = reconnect
sasl_key = ""
sasl_mechanism = plain
sasl_password = ""
sasl_timeout = 15
sasl_username = ""
split_msg_max_length = 512
ssl = off
ssl_cert = ""
ssl_dhkey_size = 2048
ssl_fingerprint = ""
ssl_password = ""
ssl_priorities = "NORMAL:-VERS-SSL3.0"
ssl_verify = on
usermode = ""
username = "'$USER'"

[server]
libera.addresses = "irc.libera.chat/7000"
libera.proxy
libera.ipv6
libera.ssl = on
libera.ssl_cert
libera.ssl_password
libera.ssl_priorities
libera.ssl_dhkey_size
libera.ssl_fingerprint
libera.ssl_verify
libera.password
libera.capabilities
libera.sasl_mechanism
libera.sasl_username
libera.sasl_password
libera.sasl_key
libera.sasl_timeout
libera.sasl_fail
libera.autoconnect
libera.autoreconnect
libera.autoreconnect_delay
libera.nicks
libera.nicks_alternate
libera.username
libera.realname
libera.local_hostname
libera.usermode
libera.command
libera.command_delay
libera.autojoin
libera.autojoin_dynamic
libera.autorejoin
libera.autorejoin_delay
libera.connection_timeout
libera.anti_flood_prio_high
libera.anti_flood_prio_low
libera.away_check
libera.away_check_max_nicks
libera.msg_kick
libera.msg_part
libera.msg_quit
libera.notify
libera.split_msg_max_length
libera.charset_message
libera.default_chantypes'>~/.config/weechat/irc.conf

# Reboot:
#--------
reboot
