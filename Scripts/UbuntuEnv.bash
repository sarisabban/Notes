#!/bin/bash

<<COMMENT
1. Run this script using the following command:
wget -O - https://raw.githubusercontent.com/sarisabban/Notes/master/Scripts/UbuntuEnv.bash | bash
2. Setup FireFox
3. Settings > Users > Add profile picture
COMMENT

# UBUNTU 22.04 ENVIRONMENT SETUP
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
#sudo sed -i "s#Icon=$HOME/.face#Icon=/usr/share/pixmaps/faces/guitar2.jpg#" /var/lib/AccountsService/users/$USER
sudo dbus-send --print-reply --system --dest=org.freedesktop.Accounts /org/freedesktop/Accounts/User"${UID}" org.freedesktop.Accounts.User.SetIconFile string:/usr/share/pixmaps/faces/guitar2.jpg

# Setup Remaining Distro Environment:
#------------------------------------
echo "[apps/update-manager]
first-run=false
launch-count=1
launch-time=int64 1709890949

[com/ubuntu/update-notifier]
release-check-time=uint32 1709890938

[org/gnome/control-center]
last-panel='ubuntu'

[org/gnome/desktop/app-folders]
folder-children=['Utilities', 'YaST']

[org/gnome/desktop/app-folders/folders/Utilities]
apps=['gnome-abrt.desktop', 'gnome-system-log.desktop', 'nm-connection-editor.desktop', 'org.gnome.baobab.desktop', 'org.gnome.Connections.desktop', 'org.gnome.DejaDup.desktop', 'org.gnome.Dictionary.desktop', 'org.gnome.DiskUtility.desktop', 'org.gnome.eog.desktop', 'org.gnome.Evince.desktop', 'org.gnome.FileRoller.desktop', 'org.gnome.fonts.desktop', 'org.gnome.seahorse.Application.desktop', 'org.gnome.tweaks.desktop', 'org.gnome.Usage.desktop', 'vinagre.desktop']
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
primary-color='#000000'
secondary-color='#000000'

[org/gnome/desktop/input-sources]
per-window=false
sources=[('xkb', 'us+mac'), ('xkb', 'ara+mac')]
xkb-options=@as []

[org/gnome/desktop/interface]
color-scheme='prefer-dark'
gtk-theme='Yaru-dark'
icon-theme='Yaru'

[org/gnome/desktop/notifications]
application-children=['org-gnome-nautilus']
show-in-lock-screen=false

[org/gnome/desktop/notifications/application/org-gnome-nautilus]
application-id='org.gnome.Nautilus.desktop'

[org/gnome/desktop/peripherals/touchpad]
tap-to-click=false
two-finger-scrolling-enabled=true

[org/gnome/desktop/privacy]
report-technical-problems=true

[org/gnome/desktop/screensaver]
color-shading-type='solid'
lock-delay=uint32 0
lock-enabled=false
picture-options='zoom'
picture-uri='file:///usr/share/backgrounds/ubuntu-wallpaper-d.png'
primary-color='#000000'
secondary-color='#000000'
ubuntu-lock-on-suspend=false

[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/evolution-data-server]
migrated=true
network-monitor-gio-name=''

[org/gnome/gedit/plugins]
active-plugins=['docinfo', 'spell', 'filebrowser', 'sort', 'openlinks', 'modelines']

[org/gnome/gedit/plugins/filebrowser]
root='file:///'
tree-view=true
virtual-root='file:///media/$HOME/Keys/OTHER%20WORK'

[org/gnome/gedit/preferences/editor]
auto-indent=false
display-line-numbers=true
highlight-current-line=false
scheme='cobalt'
tabs-size=uint32 4
wrap-last-split-mode='word'
wrap-mode='none'

[org/gnome/gedit/preferences/ui]
show-tabs-mode='auto'
statusbar-visible=false

[org/gnome/gedit/state/window]
bottom-panel-size=140
side-panel-active-page='GeditWindowDocumentsPanel'
side-panel-size=200
size=(648, 694)
state=76672

[org/gnome/nautilus/list-view]
use-tree-view=true

[org/gnome/nautilus/preferences]
default-folder-viewer='list-view'
search-filter-time-type='last_modified'

[org/gnome/nautilus/window-state]
initial-size=(890, 550)
maximized=false

[org/gnome/settings-daemon/plugins/media-keys]
custom-keybindings=['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/', '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom1/', '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom2/']

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0]
binding='<Primary><Alt>w'
command='firefox'
name='FireFox'

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom1]
binding='<Primary><Alt>h'
command==
name='Home'

[org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom2]
binding='<Primary><Alt>g'
command='gnome-text-editor'
name='Text'

[org/gnome/settings-daemon/plugins/power]
ambient-enabled=false
idle-dim=false
power-saver-profile-on-low-battery=false
sleep-inactive-ac-timeout=3600
sleep-inactive-ac-type='nothing'
sleep-inactive-battery-type='nothing'

[org/gnome/shell]
favorite-apps=['firefox_firefox.desktop', 'org.gnome.Nautilus.desktop', 'org.gnome.gedit.desktop', 'org.gnome.Terminal.desktop']
remember-mount-password=false
welcome-dialog-last-shown-version='42.9'

[org/gnome/shell/extensions/dash-to-dock]
dash-max-icon-size=48

[org/gnome/shell/extensions/ding]
show-home=false

[org/gnome/system/location]
enabled=false

[org/gnome/terminal/legacy/profiles:/:b1dcc9dd-5262-4d8d-a863-c897e6d979b9]
background-color='rgb(0,0,0)'
foreground-color='rgb(0,255,0)'
palette=['rgb(0,0,0)', 'rgb(170,0,0)', 'rgb(0,170,0)', 'rgb(170,85,0)', 'rgb(0,0,170)', 'rgb(170,0,170)', 'rgb(0,170,170)', 'rgb(170,170,170)', 'rgb(85,85,85)', 'rgb(255,85,85)', 'rgb(85,255,85)', 'rgb(255,255,85)', 'rgb(85,85,255)', 'rgb(255,85,255)', 'rgb(85,255,255)', 'rgb(255,255,255)']
scrollbar-policy='never'
use-theme-colors=false

[system/locale]
region='en_GB.UTF-8'">dconf-settings.ini
printf '%s\n' '/command==/' d i 'command="sh -c '\''setsid xdg-open '"$HOME"' &'\''"' . w q | ed -s dconf-settings.ini
#dconf dump / > dconf-settings.ini
cat dconf-settings.ini | dconf load /
rm dconf-settings.ini

# Reboot:
#--------
reboot
