#!/bin/bash

<<COMMENT
1. Run this script using the following command:
wget -O - https://raw.githubusercontent.com/sarisabban/Notes/master/Scripts/UbuntuEnv_24.04.sh | bash
2. Setup FireFox
COMMENT

# UBUNTU 24.04 ENVIRONMENT SETUP
#===============================

# Install Applications:
#----------------------
sudo apt update
sudo apt full-upgrade -y
sudo apt install weechat vim openconnect tmux htop git ffmpeg python3-venv -y

# Setup Terminal Environment:
#----------------------------
n="$(awk '/PS1=/ { $0 = "PS1=\x27\\[\\033[0;33m\\][\\W]\\$ \\[\\033[0m\\]\x27" } 1' .bashrc)"; echo "$n" > ~/.bashrc
printf ':set shiftwidth=4 noexpandtab\n:set softtabstop=4 noexpandtab\n:set list\n:set listchars=tab:»·,trail:·,extends:>,precedes:<,nbsp:+\n:syntax enable\n:set nowrap\n:set number\n:set ts=4 sw=4\n:set cc=81' >> ~/.vimrc
printf "set -g status-bg black\nset -g status-fg red\nset -g pane-border-style fg=magenta\nset -g pane-active-border-style fg=magenta\nset -g status-justify 'centre'\nset -g status-left-length '100'\nset -g status 'on'\nset -g status-right-length '100'\nsetw -g window-status-separator ''\nset -g status-right '%%H:%%M'\nsetw -g window-status-current-format '#(whoami)'\nset -g status off" > ~/.tmux.conf

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

# Turn Off Bluetooth:
#--------------------
rfkill block bluetooth

# Change User Profile Picture:
#-----------------------------
sudo dbus-send --print-reply --system --dest=org.freedesktop.Accounts /org/freedesktop/Accounts/User"${UID}" org.freedesktop.Accounts.User.SetIconFile string:/usr/share/pixmaps/faces/guitar2.jpg

# Setup Remaining Distro Environment:
#------------------------------------
echo "[com/ubuntu/update-notifier]
release-check-time=uint32 1715323391

[org/gnome/TextEditor]
auto-indent=false
last-save-directory='file:///home/atreo/Desktop'
restore-session=false
show-line-numbers=true
style-scheme='cobalt-light'
tab-width=uint32 4
use-system-font=true
wrap-text=false

[org/gnome/control-center]
last-panel='system'
window-state=(980, 640, false)

[org/gnome/desktop/app-folders]
folder-children=['Utilities', 'YaST', 'Pardus']

[org/gnome/desktop/app-folders/folders/Pardus]
categories=['X-Pardus-Apps']
name='X-Pardus-Apps.directory'
translate=true

[org/gnome/desktop/app-folders/folders/Utilities]
apps=['gnome-abrt.desktop', 'gnome-system-log.desktop', 'nm-connection-editor.desktop', 'org.gnome.baobab.desktop', 'org.gnome.Connections.desktop', 'org.gnome.DejaDup.desktop', 'org.gnome.Dictionary.desktop', 'org.gnome.DiskUtility.desktop', 'org.gnome.Evince.desktop', 'org.gnome.FileRoller.desktop', 'org.gnome.fonts.desktop', 'org.gnome.Loupe.desktop', 'org.gnome.seahorse.Application.desktop', 'org.gnome.tweaks.desktop', 'org.gnome.Usage.desktop', 'vinagre.desktop']
categories=['X-GNOME-Utilities']
name='X-GNOME-Utilities.directory'
translate=true

[org/gnome/desktop/app-folders/folders/YaST]
categories=['X-SuSE-YaST']
name='suse-yast.directory'
translate=true

[org/gnome/desktop/background]
color-shading-type='solid'
picture-options='zoom'
picture-uri='file:///usr/share/backgrounds/ubuntu-wallpaper-d.png'
picture-uri-dark='file:///usr/share/backgrounds/ubuntu-wallpaper-d.png'
primary-color='#2c001e'
secondary-color='#2c001e'

[org/gnome/desktop/datetime]
automatic-timezone=false

[org/gnome/desktop/input-sources]
mru-sources=[('xkb', 'us')]
sources=[('xkb', 'us'), ('xkb', 'ara+mac')]
xkb-options=@as []

[org/gnome/desktop/interface]
clock-show-date=false
color-scheme='prefer-dark'
gtk-theme='Yaru-dark'
icon-theme='Yaru'
show-battery-percentage=false

[org/gnome/desktop/notifications]
application-children=['gnome-power-panel', 'org-gnome-texteditor']
show-in-lock-screen=false

[org/gnome/desktop/notifications/application/gnome-power-panel]
application-id='gnome-power-panel.desktop'

[org/gnome/desktop/notifications/application/io-snapcraft-sessionagent]
enable=false

[org/gnome/desktop/notifications/application/nm-applet]
enable=false

[org/gnome/desktop/notifications/application/org-gnome-clocks]
enable=false

[org/gnome/desktop/notifications/application/org-gnome-evolution-alarm-notify]
enable=false

[org/gnome/desktop/notifications/application/org-gnome-nautilus]
enable=false

[org/gnome/desktop/notifications/application/org-gnome-texteditor]
application-id='org.gnome.TextEditor.desktop'

[org/gnome/desktop/notifications/application/org-gnome-zenity]
enable=false

[org/gnome/desktop/peripherals/touchpad]
tap-to-click=false
two-finger-scrolling-enabled=true

[org/gnome/desktop/privacy]
old-files-age=uint32 30
recent-files-max-age=-1
remember-recent-files=false

[org/gnome/desktop/screensaver]
color-shading-type='solid'
lock-enabled=false
picture-options='zoom'
picture-uri='file:///usr/share/backgrounds/ubuntu-wallpaper-d.png'
primary-color='#2c001e'
secondary-color='#2c001e'
ubuntu-lock-on-suspend=false

[org/gnome/desktop/search-providers]
sort-order=['org.gnome.Contacts.desktop', 'org.gnome.Documents.desktop', 'org.gnome.Nautilus.desktop']

[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/desktop/sound]
allow-volume-above-100-percent=false
event-sounds=true

[org/gnome/evolution-data-server]
migrated=true

[org/gnome/mutter]
edge-tiling=false

[org/gnome/mutter/keybindings]
toggle-tiled-left=@as []
toggle-tiled-right=@as []

[org/gnome/nautilus/preferences]
default-folder-viewer='list-view'
migrated-gtk-settings=true
search-filter-time-type='last_modified'

[org/gnome/nautilus/window-state]
initial-size=(890, 550)

[org/gnome/settings-daemon/plugins/color]
night-light-enabled=true
night-light-temperature=uint32 2700

[org/gnome/settings-daemon/plugins/media-keys]
custom-keybindings=['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/', '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom1/', '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom2/']

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0]
binding='<Control><Alt>w'
command='firefox'
name='Web Browser'

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom1]
binding='<Control><Alt>h'
command="sh -c 'setsid xdg-open \"$HOME\" &'"
name='Home'

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom2]
binding='<Control><Alt>g'
command='gnome-text-editor'
name='Text'

[org/gnome/settings-daemon/plugins/power]
ambient-enabled=false
idle-dim=false
power-saver-profile-on-low-battery=false
sleep-inactive-ac-timeout=3600
sleep-inactive-ac-type='nothing'

[org/gnome/shell]
app-picker-layout=[{'org.gnome.Terminal.desktop': <{'position': <0>}>, 'software-properties-drivers.desktop': <{'position': <1>}>, 'firmware-updater_firmware-updater.desktop': <{'position': <2>}>, 'org.gnome.eog.desktop': <{'position': <3>}>, 'org.gnome.clocks.desktop': <{'position': <4>}>, 'gnome-language-selector.desktop': <{'position': <5>}>, 'org.gnome.PowerStats.desktop': <{'position': <6>}>, 'software-properties-gtk.desktop': <{'position': <7>}>, 'update-manager.desktop': <{'position': <8>}>, 'org.gnome.Calculator.desktop': <{'position': <9>}>, 'gnome-session-properties.desktop': <{'position': <10>}>, 'org.gnome.TextEditor.desktop': <{'position': <11>}>, 'org.gnome.Settings.desktop': <{'position': <12>}>, 'org.gnome.SystemMonitor.desktop': <{'position': <13>}>, 'Utilities': <{'position': <14>}>, 'yelp.desktop': <{'position': <15>}>, 'snap-store_snap-store.desktop': <{'position': <16>}>}]
enabled-extensions=['ding@rastersoft.com', 'ubuntu-dock@ubuntu.com', 'tiling-assistant@ubuntu.com']
favorite-apps=['firefox_firefox.desktop', 'org.gnome.Nautilus.desktop', 'org.gnome.TextEditor.desktop', 'org.gnome.Terminal.desktop']
welcome-dialog-last-shown-version='46.0'

[org/gnome/shell/app-switcher]
current-workspace-only=false

[org/gnome/shell/extensions/dash-to-dock]
dash-max-icon-size=48
extend-height=true
isolate-workspaces=false

[org/gnome/shell/extensions/ding]
check-x11wayland=true
show-home=false
start-corner='top-right'

[org/gnome/shell/extensions/tiling-assistant]
active-window-hint-color='rgb(211,70,21)'
last-version-installed=46
overridden-settings={'org.gnome.mutter.edge-tiling': <@mb nothing>, 'org.gnome.mutter.keybindings.toggle-tiled-left': <@mb nothing>, 'org.gnome.mutter.keybindings.toggle-tiled-right': <@mb nothing>}
tiling-popup-all-workspace=true

[org/gnome/shell/world-clocks]
locations=@av []

[org/gnome/terminal/legacy/profiles:/:b1dcc9dd-5262-4d8d-a863-c897e6d979b9]
background-color='rgb(0,0,0)'
foreground-color='rgb(0,255,0)'
palette=['rgb(0,0,0)', 'rgb(170,0,0)', 'rgb(0,170,0)', 'rgb(170,85,0)', 'rgb(0,0,170)', 'rgb(170,0,170)', 'rgb(0,170,170)', 'rgb(170,170,170)', 'rgb(85,85,85)', 'rgb(255,85,85)', 'rgb(85,255,85)', 'rgb(255,255,85)', 'rgb(85,85,255)', 'rgb(255,85,255)', 'rgb(85,255,255)', 'rgb(255,255,255)']
scrollbar-policy='never'
use-theme-colors=false

[org/gtk/gtk4/settings/file-chooser]
date-format='regular'
location-mode='path-bar'
show-hidden=true
sidebar-width=140
sort-column='name'
sort-directories-first=true
sort-order='ascending'
type-format='category'
view-type='list'
window-size=(1192, 372)

[system/locale]
region='en_GB.UTF-8'">dconf-settings.ini
printf '%s\n' '/command==/' d i 'command="sh -c '\''setsid xdg-open '"$HOME"' &'\''"' . w q | ed -s dconf-settings.ini
cat dconf-settings.ini | dconf load / # Dump info using dconf dump / > dconf-settings.ini
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
# For more info, see: https://weechat.org/doc/weechat/quickstart/
#

config_version = 3

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
display_host_notice = on
display_host_quit = on
display_join_message = "329,332,333,366"
display_old_topic = on
display_pv_away_once = on
display_pv_back = on
display_pv_nick_change = on
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
list_buffer_scroll_horizontal = 10
list_buffer_sort = "~name2"
list_buffer_topic_strip_colors = on
msgbuffer_fallback = current
new_channel_position = none
new_list_position = none
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
open_pv_buffer_echo_msg = on
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
smart_filter_setname = on
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
list_buffer_line_selected = white
list_buffer_line_selected_bg = 24
message_account = cyan
message_chghost = brown
message_join = green
message_kick = red
message_quit = red
message_setname = brown
mirc_remap = "1,-1:darkgray"
nick_prefixes = "y:lightred;q:lightred;a:lightcyan;o:lightgreen;h:lightmagenta;v:yellow;*:lightblue"
notice = green
reason_kick = default
reason_quit = 244
topic_current = default
topic_new = 36
topic_old = 244

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
clientinfo = "${clientinfo}"
source = "${download}"
time = "${time}"
version = "WeeChat ${version}"

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
registered_mode = "r"
sasl_fail = reconnect
sasl_key = ""
sasl_mechanism = plain
sasl_password = ""
sasl_timeout = 15
sasl_username = ""
split_msg_max_length = 512
tls = on
tls_cert = ""
tls_dhkey_size = 2048
tls_fingerprint = ""
tls_password = ""
tls_priorities = "NORMAL:-VERS-SSL3.0"
tls_verify = on
usermode = ""
username = "'$USER'"

[server]
libera.addresses = "irc.libera.chat/7000"
libera.proxy
libera.ipv6
libera.tls
libera.tls_cert
libera.tls_password
libera.tls_priorities
libera.tls_dhkey_size
libera.tls_fingerprint
libera.tls_verify
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
libera.default_chantypes
libera.registered_mode'>~/.config/weechat/irc.conf

# Reboot:
#--------
reboot
